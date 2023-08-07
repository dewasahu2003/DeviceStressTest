use tch::Tensor;

pub fn test() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let device = match args
        .iter()
        .map(|x| x.as_str())
        .collect::<Vec<_>>()
        .as_slice()
    {
        [_] => tch::Device::Cpu,
        [_, "cpu"] => tch::Device::Cpu,
        [_, "gpu"] => tch::Device::Cuda(0),
        _ => panic!("usage main cpu|gpu"),
    };
    let slice = vec![1; 1_000_000];
    for i in 1..1_000_000 {
        let t = Tensor::of_slice(&slice).to_device(device);
        println!("{} {:?}", i, t.size());
    }
    Ok(())
}
