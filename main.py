import argparse
import torch
from data_loader import get_data_loaders
from model import get_model
from trainer import Trainer
from predictor import LegalSpellingCorrector


def main(args):
    # 检测可用设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    # 数据加载
    print("加载数据...")
    train_loader, test_loader, tokenizer = get_data_loaders(
        train_path=args.train_file,
        test_path=args.test_file,
        batch_size=args.batch_size,
        max_len=args.max_len
    )

    # 如果需要训练
    if args.mode == 'train' or args.mode == 'all':
        print("初始化模型...")
        model = get_model(device=device)

        print("开始训练...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            tokenizer=tokenizer,
            epochs=args.epochs,
            save_dir=args.save_dir,
            device=device
        )

        # 执行训练
        trainer.train()

    # 如果需要预测
    if args.mode == 'predict' or args.mode == 'all':
        print("加载模型进行预测...")
        corrector = LegalSpellingCorrector(
            model_path=args.save_dir + '/best_model',
            device=device
        )

        # 测试一些例子
        test_examples = [
            "行政机关实施行政管理都应当公开，这是程序正档原则的要求",
            "对于指控被告人犯数罪的案件，如果被告人不分人罪，部分不认醉，则斗不能适用",
            "当事人订立合同，可以采取要约、承若方式或者其他方式"
        ]

        print("\n预测结果:")
        print("-" * 50)
        for example in test_examples:
            corrected = corrector.correct_text(example)
            comparison = corrector.compare_text(example, corrected)

            print(f"原始文本: {example}")
            print(f"纠正文本: {corrected}")

            if comparison['differences']:
                print("修改点:")
                for diff in comparison['differences']:
                    print(f"  {diff['original']} → {diff['corrected']}")
            else:
                print("  无修改")
            print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="法律文书拼写纠错系统（GPU支持）")

    # 数据参数
    parser.add_argument('--train_file', type=str, default='law.train', help='训练数据集路径')
    parser.add_argument('--test_file', type=str, default='law.test', help='测试数据集路径')
    parser.add_argument('--max_len', type=int, default=128, help='文本最大长度')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--save_dir', type=str, default='models', help='模型保存目录')
    parser.add_argument('--force_cpu', action='store_true', help='强制使用CPU')

    # 运行模式
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'all'],
                        default='all', help='运行模式：train(仅训练), predict(仅预测), all(全部)')

    args = parser.parse_args()
    main(args)
