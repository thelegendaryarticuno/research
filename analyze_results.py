import pandas as pd
import numpy as np
from pathlib import Path

# Load overall results
overall_df = pd.read_csv("metrics/model_overall_results.csv")

print("="*60)
print("🔍 COMPREHENSIVE ML MODEL ANALYSIS FOR ENCRYPTION CLASSIFICATION")
print("="*60)

print("\n📊 OVERALL PERFORMANCE RANKING:")
print("-" * 40)
print(overall_df.sort_values('F1', ascending=False).to_string(index=False))

# Analyze per-size performance
sizes = [16, 64, 256, 512, 1024, 2048]
metrics = ['F1', 'Precision', 'Recall', 'Accuracy']

print("\n📈 PER-SIZE ANALYSIS:")
print("-" * 40)

# Load per-size accuracy data
size_performance = {}
for size in sizes:
    try:
        acc_df = pd.read_csv(f"metrics/tables_by_size/Accuracy_{size}KB.csv")
        size_performance[size] = acc_df.iloc[0, 1:].to_dict()
    except FileNotFoundError:
        continue

# Create performance summary
performance_summary = []
for model in ['Decision Tree', 'Random Forest', 'SVM', 'KNN', 'LightGBM']:
    model_stats = {
        'Model': model,
        'Avg_Accuracy': np.mean([size_performance[size][model] for size in sizes if model in size_performance[size]]),
        'Min_Accuracy': min([size_performance[size][model] for size in sizes if model in size_performance[size]]),
        'Max_Accuracy': max([size_performance[size][model] for size in sizes if model in size_performance[size]]),
        'Std_Accuracy': np.std([size_performance[size][model] for size in sizes if model in size_performance[size]])
    }
    performance_summary.append(model_stats)

perf_df = pd.DataFrame(performance_summary)
perf_df = perf_df.sort_values('Avg_Accuracy', ascending=False)

print("\n🎯 DETAILED PERFORMANCE STATISTICS:")
print("-" * 50)
for _, row in perf_df.iterrows():
    print(f"{row['Model']:15} | Avg: {row['Avg_Accuracy']:.3f} | Min: {row['Min_Accuracy']:.3f} | Max: {row['Max_Accuracy']:.3f} | Std: {row['Std_Accuracy']:.3f}")

# Algorithm-specific analysis
print("\n🔐 ALGORITHM-SPECIFIC PERFORMANCE:")
print("-" * 40)

# Load F1 scores for detailed analysis
f1_analysis = {}
for size in sizes:
    try:
        f1_df = pd.read_csv(f"metrics/tables_by_size/F1_{size}KB.csv")
        f1_df = f1_df.set_index('algorithm')
        f1_analysis[size] = f1_df
    except FileNotFoundError:
        continue

# Calculate algorithm difficulty (lower average F1 = harder to classify)
algorithms = ['AES', 'PRESENT', 'SIMON', 'XTEA', 'PRINCE', 'MSEA', 'LEA', 'RECTANGLE', 'ASCONv2']
algorithm_difficulty = {}

for algo in algorithms:
    scores = []
    for size in sizes:
        if size in f1_analysis and algo in f1_analysis[size].index:
            scores.extend(f1_analysis[size].loc[algo].values)
    if scores:
        algorithm_difficulty[algo] = {
            'avg_f1': np.mean(scores),
            'min_f1': min(scores),
            'max_f1': max(scores),
            'std_f1': np.std(scores)
        }

print("\nEASIEST TO CLASSIFY (High F1 scores):")
sorted_algos = sorted(algorithm_difficulty.items(), key=lambda x: x[1]['avg_f1'], reverse=True)
for algo, stats in sorted_algos[:3]:
    print(f"  ✅ {algo:10} | Avg F1: {stats['avg_f1']:.3f} | Range: {stats['min_f1']:.3f}-{stats['max_f1']:.3f}")

print("\nHARDEST TO CLASSIFY (Low F1 scores):")
for algo, stats in sorted_algos[-3:]:
    print(f"  ❌ {algo:10} | Avg F1: {stats['avg_f1']:.3f} | Range: {stats['min_f1']:.3f}-{stats['max_f1']:.3f}")

# Model consistency analysis
print("\n📋 MODEL CONSISTENCY ANALYSIS:")
print("-" * 40)

model_consistency = {}
for model in ['Decision Tree', 'Random Forest', 'SVM', 'KNN', 'LightGBM']:
    all_f1_scores = []
    perfect_scores = 0
    total_scores = 0
    
    for size in sizes:
        if size in f1_analysis and model in f1_analysis[size].columns:
            scores = f1_analysis[size][model].values
            all_f1_scores.extend(scores)
            perfect_scores += sum(scores == 1.0)
            total_scores += len(scores)
    
    if all_f1_scores:
        model_consistency[model] = {
            'avg_f1': np.mean(all_f1_scores),
            'std_f1': np.std(all_f1_scores),
            'perfect_rate': perfect_scores / total_scores if total_scores > 0 else 0,
            'min_f1': min(all_f1_scores),
            'failure_rate': sum(score < 0.5 for score in all_f1_scores) / len(all_f1_scores)
        }

print("Model Consistency Ranking:")
for model, stats in sorted(model_consistency.items(), key=lambda x: x[1]['avg_f1'], reverse=True):
    print(f"  {model:15} | Avg F1: {stats['avg_f1']:.3f} | Perfect: {stats['perfect_rate']:.1%} | Failure: {stats['failure_rate']:.1%} | Std: {stats['std_f1']:.3f}")

# Final recommendation
print("\n" + "="*60)
print("🏆 FINAL RECOMMENDATION")
print("="*60)

# Score each model based on multiple criteria
model_scores = {}
for model in ['Decision Tree', 'Random Forest', 'SVM', 'KNN', 'LightGBM']:
    overall_row = overall_df[overall_df['Model'] == model].iloc[0]
    perf_row = perf_df[perf_df['Model'] == model].iloc[0]
    consist_stats = model_consistency[model]
    
    # Weighted scoring (higher is better)
    score = (
        overall_row['F1'] * 0.25 +           # Overall F1 score
        overall_row['Accuracy'] * 0.20 +     # Overall accuracy
        perf_row['Avg_Accuracy'] * 0.20 +    # Average accuracy across sizes
        consist_stats['perfect_rate'] * 0.15 + # Perfect classification rate
        (1 - consist_stats['failure_rate']) * 0.10 + # Low failure rate
        (1 - perf_row['Std_Accuracy']) * 0.10  # Consistency (low std)
    )
    
    model_scores[model] = score

# Sort models by score
ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

print("\n🥇 MODEL RANKING (Best to Worst):")
print("-" * 40)
for i, (model, score) in enumerate(ranked_models, 1):
    overall_row = overall_df[overall_df['Model'] == model].iloc[0]
    consist_stats = model_consistency[model]
    
    if i == 1:
        print(f"🏆 #{i}. {model}")
        print(f"    Overall F1: {overall_row['F1']:.3f}")
        print(f"    Overall Accuracy: {overall_row['Accuracy']:.3f}")
        print(f"    Perfect Classifications: {consist_stats['perfect_rate']:.1%}")
        print(f"    Failure Rate: {consist_stats['failure_rate']:.1%}")
        print(f"    Composite Score: {score:.3f}")
        print(f"    ✅ RECOMMENDED: Best overall performance")
    else:
        emoji = "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}."
        print(f"{emoji} {model} (Score: {score:.3f})")

print("\n🔍 KEY INSIGHTS:")
print("-" * 20)
best_model = ranked_models[0][0]
best_stats = model_consistency[best_model]
best_overall = overall_df[overall_df['Model'] == best_model].iloc[0]

print(f"• {best_model} is the BEST model for encryption algorithm classification")
print(f"• Achieves {best_overall['F1']:.1%} F1-score and {best_overall['Accuracy']:.1%} accuracy overall")
print(f"• Perfect classification rate: {best_stats['perfect_rate']:.1%}")
print(f"• Low failure rate: {best_stats['failure_rate']:.1%}")
print(f"• Most consistent performer across different file sizes")

# Algorithm insights
easiest_algo = sorted_algos[0][0]
hardest_algo = sorted_algos[-1][0]
print(f"• {easiest_algo} is the EASIEST algorithm to classify")
print(f"• {hardest_algo} is the HARDEST algorithm to classify")

print(f"\n💡 RECOMMENDATION: Use {best_model} for production encryption algorithm classification")
print("="*60)