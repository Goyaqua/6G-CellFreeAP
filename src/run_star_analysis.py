"""
Easy-to-use script for analyzing trained models on Star topology
Automatically finds and analyzes all models in results directories
"""

import argparse
from pathlib import Path
import glob
from analyze_star_topology_heatmap import analyze_multiple_models, StarTopologyAnalyzer


def find_trained_models(results_dir: str = ".", pattern: str = "**/*best_model.zip") -> list:
    """
    Find all trained models in results directory

    Args:
        results_dir: Root directory to search
        pattern: Glob pattern for model files

    Returns:
        List of model paths
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Directory not found: {results_dir}")
        return []

    # Search for models
    model_paths = list(results_path.glob(pattern))

    # Also try alternative patterns
    if not model_paths:
        model_paths = list(results_path.glob("**/*.zip"))

    if not model_paths:
        print(f"No models found in {results_dir}")
        print(f"Searched pattern: {pattern}")
        return []

    # Sort by path
    model_paths = sorted(model_paths)

    print(f"Found {len(model_paths)} models:")
    for i, path in enumerate(model_paths, 1):
        print(f"  {i}. {path}")

    return [str(p) for p in model_paths]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trained RL models on Star (cross) topology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single model
  python run_star_analysis.py --model path/to/model.zip

  # Analyze all models in a directory
  python run_star_analysis.py --dir results_unified_reward_20260111_235805

  # Analyze with custom settings
  python run_star_analysis.py --dir results --num-episodes 20 --episode-length 200

  # Specify output directory
  python run_star_analysis.py --dir results --output star_analysis_output
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Path to single model file'
    )

    parser.add_argument(
        '--dir',
        type=str,
        default='.',
        help='Directory to search for models (default: current directory)'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='**/*best_model.zip',
        help='Glob pattern for finding models (default: **/*best_model.zip)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='star_topology_analysis',
        help='Output directory for results (default: star_topology_analysis)'
    )

    parser.add_argument(
        '--num-aps',
        type=int,
        default=64,
        help='Number of APs (default: 64)'
    )

    parser.add_argument(
        '--num-users',
        type=int,
        default=16,
        help='Number of users (default: 16)'
    )

    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
        help='Number of test episodes (default: 10)'
    )

    parser.add_argument(
        '--episode-length',
        type=int,
        default=100,
        help='Steps per episode (default: 100)'
    )

    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only list found models without analyzing'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STAR TOPOLOGY ANALYSIS TOOL")
    print("=" * 80)

    # Determine which models to analyze
    if args.model:
        # Single model specified
        model_paths = [args.model]
        if not Path(args.model).exists():
            print(f"Error: Model not found: {args.model}")
            return
        print(f"Analyzing single model: {args.model}")
    else:
        # Search directory for models
        print(f"Searching for models in: {args.dir}")
        print(f"Pattern: {args.pattern}\n")
        model_paths = find_trained_models(args.dir, args.pattern)

        if not model_paths:
            print("\nNo models found. Please check the directory path and pattern.")
            return

    if args.list_only:
        print("\n--list-only specified, exiting without analysis")
        return

    # Confirm before running
    print(f"\nSettings:")
    print(f"  Number of APs: {args.num_aps}")
    print(f"  Number of Users: {args.num_users}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Episode Length: {args.episode_length}")
    print(f"  Output Directory: {args.output}")

    user_input = input("\nProceed with analysis? [Y/n]: ")
    if user_input.lower() not in ['', 'y', 'yes']:
        print("Analysis cancelled.")
        return

    # Run analysis
    if len(model_paths) == 1:
        # Single model
        print("\n" + "=" * 80)
        print("ANALYZING SINGLE MODEL")
        print("=" * 80)

        analyzer = StarTopologyAnalyzer(
            model_path=model_paths[0],
            num_aps=args.num_aps,
            num_users=args.num_users,
            num_episodes=args.num_episodes,
            episode_length=args.episode_length
        )

        results = analyzer.run_analysis()
        analyzer.visualize_heatmaps(results, save_dir=args.output)
        analyzer.save_results(results, save_dir=args.output)

    else:
        # Multiple models
        print("\n" + "=" * 80)
        print(f"ANALYZING {len(model_paths)} MODELS")
        print("=" * 80)

        analyze_multiple_models(model_paths, output_dir=args.output)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
