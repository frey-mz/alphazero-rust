
use burn::{backend::{wgpu::WgpuDevice, Wgpu}, nn::BatchNorm, prelude::*, tensor::activation::log_softmax};
use nn::{conv::{Conv2d}, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu};

fn main(){
    let net = Connect4Net::new();
    let (x, y) = net.forward(Tensor::zeros([1,2,6,7], &WgpuDevice::BestAvailable));
    println!("{x} {y}");
    
}


#[derive(Clone, Module, Debug)]
struct Connect4Net {
    conv: ConvBlock<Wgpu>,
    outblock: OutBlock<Wgpu>,
    resblocks: [ResBlock<Wgpu>;5]
}

impl Connect4Net {
    fn new() -> Self {
        let device = &WgpuDevice::BestAvailable;
        Self {
            conv: ConvBlock::new([2, 128], [3,3], device),
            outblock: OutBlock::new(device),
            resblocks: std::array::from_fn(|_|ResBlock::new(device)),
        }
    }
    fn forward(&self, input: Tensor<Wgpu, 4>)-> (Tensor<Wgpu, 1>, Tensor<Wgpu, 1>){

        let mut input = self.conv.forward(input);
        for block in self.resblocks.iter(){
            input = block.forward(input);
        }
        self.outblock.forward(input)

    }
}




#[derive(Module, Debug)]
struct ResBlock<B: Backend> {
    conv:  ConvBlock<B>,
    conv2: ConvBlock<B>,

}
impl<B: Backend> ResBlock<B> {
    fn new(device: &Device<B>) -> Self {

        Self {
            conv: ConvBlock::new([128, 128], [3,3], device),
            conv2: ConvBlock::new([128, 128], [3, 3], device),
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = input.clone();
        let input = self.conv.forward(input);
        let input = self.conv2.forward_with_residual(input, residual);

        input
    }
}



#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B, 2>,
    activation: Relu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(channels: [usize; 2], kernel_size: [usize; 2], device: &B::Device) -> Self {
        let conv = nn::conv::Conv2dConfig::new(channels, kernel_size)
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let norm = BatchNormConfig::new(channels[1]).init(device);

        Self {
            conv,
            norm,
            activation: nn::Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);

        self.activation.forward(x)
    }
    pub fn forward_with_residual(&self, input: Tensor<B, 4>, residual: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        let x = x.add(residual);

        self.activation.forward(x)
    }
}


#[derive(Module, Debug)]
struct OutBlock<B: Backend> {
    conv: ConvBlock<B>,
    conv2: ConvBlock<B>,
    fc: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,

}
impl<B: Backend> OutBlock<B> {
    fn new(device: &B::Device) -> Self {

        Self {
            conv: ConvBlock::new([2, 128], [3, 3], device),
            conv2: ConvBlock::new([32, 128], [3, 3], device),
            fc: LinearConfig::new(6*7*32, 7).init(device),
            fc2: LinearConfig::new(2*6*7, 32).init(device),
            fc3: LinearConfig::new(32, 1).init(device),

        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let value = self.conv.forward(input.clone());

        let value = value.reshape([2*6*7_i32]);
        let value = self.fc2.forward(value);
        let value = Relu::new().forward(value);


        let value = self.fc3.forward(value);
        let value = value.tanh();

        let policy = self.conv2.forward(input);

        let policy = policy.reshape([6 * 7 * 32_i32]);
        let policy = self.fc.forward(policy);

        let policy = log_softmax(policy, 0).exp();

        (policy, value)
    }
}
