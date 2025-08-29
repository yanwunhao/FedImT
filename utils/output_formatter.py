import numpy as np

def print_header(args):
    """Print formatted header with configuration info"""
    print("=" * 80)
    print("FEDERATED LEARNING WITH IMBALANCE-AWARE MONITORING")
    print("=" * 80)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Model: {args.model.upper()}")
    print(f"Rounds: {args.rounds}")
    print(f"Clients: {args.num_users}")
    print(f"Fraction: {args.frac}")
    print(f"Device: {args.device}")
    print("-" * 80)

def print_progress(current_round, total_rounds):
    """Print progress bar for training rounds"""
    progress = (current_round + 1) / total_rounds
    bar_length = 40
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '▒' * (bar_length - filled_length)
    percentage = progress * 100
    print(f"\nProgress: [{bar}] {percentage:.1f}% (Round {current_round + 1}/{total_rounds})")

def print_round_summary(round_num, loss_avg, ac_avg, acc_test, loss_test, T_j, T_G, selected_composition):
    """Print comprehensive round summary"""
    print(f"\n{'='*20} ROUND {round_num + 1} SUMMARY {'='*20}")
    print(f"Selected Clients Composition: {np.array(selected_composition).round(3)}")
    print(f"Training - Loss: {loss_avg:.4f} | Accuracy: {ac_avg:.4f}")
    print(f"Testing  - Loss: {loss_test:.4f} | Accuracy: {acc_test:.4f}")
    print(f"Metrics  - T_j: {T_j:.4f} | T_G: {T_G:.4f}")
    print("-" * 60)

def print_final_results(Tj_buffer, TG_buffer):
    """Print final experiment results"""
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"Results saved to:")
    print(f"  - T_j values: results/Tj_1.txt")
    print(f"  - T_G values: results/TG_1.txt")
    print(f"Average T_j: {np.mean([float(x.strip()) for x in Tj_buffer]):.4f}")
    print(f"Average T_G: {np.mean([float(x.strip()) for x in TG_buffer]):.4f}")
    print("=" * 80)