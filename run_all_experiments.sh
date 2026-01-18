#!/bin/bash

################################################################################
# Unified Reward Function - Complete Training & Analysis Pipeline
#
# Compatible with Bash 3.2+ (macOS default)
#
# This script:
# 1. Trains all 11 scenarios with the new unified reward function
# 2. Runs performance analysis (analyze_network.py) for each trained model
# 3. Runs behavior analysis (analyze_behavior.py) for each trained model
# 4. Saves all results, plots, and logs in organized directories
#
# Usage: bash run_all_experiments.sh
################################################################################

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_RESULTS_DIR="./results_unified_reward_${TIMESTAMP}"
TENSORBOARD_DIR="./tensorboard_unified_${TIMESTAMP}"

# Create base directories
mkdir -p "${BASE_RESULTS_DIR}"
mkdir -p "${TENSORBOARD_DIR}"

################################################################################
# Scenario Data (Compatible with Bash 3.2)
################################################################################

# Function to get scenario config path
get_config_path() {
    case $1 in
        1) echo "configs/ppo_scenarios/1_sweet_spot_balanced.yaml" ;;
        2) echo "configs/ppo_scenarios/2_fix_rate_balanced_25ap.yaml" ;;
        3) echo "configs/ppo_scenarios/3_scalability_high_int.yaml" ;;
        4) echo "configs/ppo_scenarios/4_massive_mimo_64ap.yaml" ;;
        5) echo "configs/ppo_scenarios/5_qos_max_speed_36ap.yaml" ;;
        6) echo "configs/ppo_scenarios/6_sparse_network_10ap.yaml" ;;
        7) echo "configs/ppo_scenarios/7_mimo_collocated_16ap_4ant copy.yaml" ;;
        8) echo "configs/ppo_scenarios/8_mimo_crowded.yaml" ;;
        9) echo "configs/ppo_scenarios/9_low_load_green.yaml" ;;
        10) echo "configs/ppo_scenarios/10_eco_mode_aggressive.yaml" ;;
        11) echo "configs/ppo_scenarios/11_performance_mode_max_rate.yaml" ;;
    esac
}

# Function to get scenario name
get_scenario_name() {
    case $1 in
        1) echo "Sweet_Spot_Balanced" ;;
        2) echo "Fix_Rate_25AP" ;;
        3) echo "Scalability_High_Interference" ;;
        4) echo "Massive_MIMO_64AP" ;;
        5) echo "QoS_Max_Speed" ;;
        6) echo "Sparse_Network_10AP" ;;
        7) echo "MIMO_Collocated_16AP_4Ant" ;;
        8) echo "MIMO_Crowded_264Users" ;;
        9) echo "Low_Load_Green" ;;
        10) echo "Eco_Mode_Aggressive" ;;
        11) echo "Performance_Mode_Max_Rate" ;;
    esac
}

# Function to get number of APs
get_num_aps() {
    case $1 in
        1) echo 36 ;;
        2) echo 25 ;;
        3) echo 36 ;;
        4) echo 64 ;;
        5) echo 36 ;;
        6) echo 10 ;;
        7) echo 16 ;;
        8) echo 64 ;;
        9) echo 36 ;;
        10) echo 36 ;;
        11) echo 36 ;;
    esac
}

# Function to get number of users
get_num_users() {
    case $1 in
        1) echo 10 ;;
        2) echo 10 ;;
        3) echo 20 ;;
        4) echo 10 ;;
        5) echo 10 ;;
        6) echo 5 ;;
        7) echo 10 ;;
        8) echo 264 ;;
        9) echo 5 ;;
        10) echo 10 ;;
        11) echo 10 ;;
    esac
}

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "\n${BLUE}============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

################################################################################
# Phase 1: Training All Scenarios
################################################################################

train_scenario() {
    local scenario_num=$1
    local config_path=$2
    local scenario_name=$3

    print_header "TRAINING SCENARIO ${scenario_num}: ${scenario_name}"

    local train_log="${BASE_RESULTS_DIR}/scenario_${scenario_num}_training.log"

    print_info "Config: ${config_path}"
    print_info "Log: ${train_log}"

    # Run training
    if python src/train_agent.py \
        --config "${config_path}" \
        --agent ppo \
        > "${train_log}" 2>&1; then

        print_success "Training completed for Scenario ${scenario_num}"

        # Find the latest experiment directory
        local latest_exp=$(ls -td experiments/exp_* 2>/dev/null | head -1)

        if [ -z "$latest_exp" ]; then
            print_error "No experiment directory found after training"
            return 1
        fi

        print_info "Model saved in: ${latest_exp}"

        # Save experiment path for later analysis
        echo "${latest_exp}" > "${BASE_RESULTS_DIR}/scenario_${scenario_num}_model_path.txt"

        return 0
    else
        print_error "Training failed for Scenario ${scenario_num}"
        print_error "Check log: ${train_log}"
        return 1
    fi
}

################################################################################
# Phase 2: Network Performance Analysis
################################################################################

analyze_network_performance() {
    local scenario_num=$1
    local scenario_name=$2
    local num_aps=$3
    local num_users=$4

    print_header "NETWORK ANALYSIS - SCENARIO ${scenario_num}: ${scenario_name}"

    # Read model path
    local model_path_file="${BASE_RESULTS_DIR}/scenario_${scenario_num}_model_path.txt"

    if [ ! -f "${model_path_file}" ]; then
        print_error "Model path file not found: ${model_path_file}"
        return 1
    fi

    local exp_dir=$(cat "${model_path_file}")
    local model_path="${exp_dir}/models/ppo_cellfree_final"

    if [ ! -f "${model_path}.zip" ]; then
        print_error "Model file not found: ${model_path}.zip"
        return 1
    fi

    local analysis_log="${BASE_RESULTS_DIR}/scenario_${scenario_num}_network_analysis.log"

    print_info "Model: ${model_path}"
    print_info "Network: ${num_aps} APs, ${num_users} Users"
    print_info "Log: ${analysis_log}"

    # Run network analysis
    if python src/analyze_network.py \
        --mode evaluate \
        --model "${model_path}" \
        --episodes 100 \
        --num-aps ${num_aps} \
        --num-users ${num_users} \
        > "${analysis_log}" 2>&1; then

        print_success "Network analysis completed for Scenario ${scenario_num}"

        # Move plots to organized directory
        local plots_dir="${BASE_RESULTS_DIR}/scenario_${scenario_num}_network_plots"
        mkdir -p "${plots_dir}"

        if [ -d "./plots" ]; then
            mv ./plots/* "${plots_dir}/" 2>/dev/null || true
            print_success "Plots saved to: ${plots_dir}"
        fi

        return 0
    else
        print_error "Network analysis failed for Scenario ${scenario_num}"
        print_error "Check log: ${analysis_log}"
        return 1
    fi
}

################################################################################
# Phase 3: Behavior Analysis
################################################################################

analyze_behavior() {
    local scenario_num=$1
    local scenario_name=$2
    local config_path=$3

    print_header "BEHAVIOR ANALYSIS - SCENARIO ${scenario_num}: ${scenario_name}"

    # Read model path
    local model_path_file="${BASE_RESULTS_DIR}/scenario_${scenario_num}_model_path.txt"

    if [ ! -f "${model_path_file}" ]; then
        print_error "Model path file not found: ${model_path_file}"
        return 1
    fi

    local exp_dir=$(cat "${model_path_file}")
    local model_path="${exp_dir}/models/ppo_cellfree_final"

    local behavior_dir="${BASE_RESULTS_DIR}/scenario_${scenario_num}_behavior"
    local analysis_log="${BASE_RESULTS_DIR}/scenario_${scenario_num}_behavior_analysis.log"

    print_info "Model: ${model_path}"
    print_info "Config: ${config_path}"
    print_info "Save Dir: ${behavior_dir}"
    print_info "Log: ${analysis_log}"

    # Run behavior analysis
    if python src/analyze_behavior.py \
        --model "${model_path}" \
        --config "${config_path}" \
        --episodes 100 \
        --save_dir "${behavior_dir}" \
        > "${analysis_log}" 2>&1; then

        print_success "Behavior analysis completed for Scenario ${scenario_num}"
        print_success "Results saved to: ${behavior_dir}"

        return 0
    else
        print_error "Behavior analysis failed for Scenario ${scenario_num}"
        print_error "Check log: ${analysis_log}"
        return 1
    fi
}

################################################################################
# Main Pipeline
################################################################################

main() {
    print_header "UNIFIED REWARD FUNCTION - COMPLETE PIPELINE"
    print_info "Timestamp: ${TIMESTAMP}"
    print_info "Results Directory: ${BASE_RESULTS_DIR}"
    print_info "Tensorboard Directory: ${TENSORBOARD_DIR}"

    # Summary counters
    local total_scenarios=11
    local trained_count=0
    local network_analysis_count=0
    local behavior_analysis_count=0

    # Track failed scenarios (using simple strings)
    local failed_training=""
    local failed_network=""
    local failed_behavior=""

    # Phase 1: Train all scenarios
    print_header "PHASE 1/3: TRAINING ALL SCENARIOS"

    for scenario_num in 1 2 3 4 5 6 7 8 9 10 11; do
        config_path=$(get_config_path $scenario_num)
        scenario_name=$(get_scenario_name $scenario_num)

        if train_scenario "${scenario_num}" "${config_path}" "${scenario_name}"; then
            trained_count=$((trained_count + 1))
        else
            failed_training="${failed_training} ${scenario_num}"
        fi

        # Small delay between trainings
        sleep 2
    done

    print_success "Training Phase Complete: ${trained_count}/${total_scenarios} succeeded"

    # Phase 2: Network Analysis
    print_header "PHASE 2/3: NETWORK PERFORMANCE ANALYSIS"

    for scenario_num in 1 2 3 4 5 6 7 8 9 10 11; do
        # Skip if training failed
        if echo "$failed_training" | grep -q " ${scenario_num}"; then
            print_warning "Skipping network analysis for Scenario ${scenario_num} (training failed)"
            continue
        fi

        scenario_name=$(get_scenario_name $scenario_num)
        num_aps=$(get_num_aps $scenario_num)
        num_users=$(get_num_users $scenario_num)

        if analyze_network_performance "${scenario_num}" "${scenario_name}" "${num_aps}" "${num_users}"; then
            network_analysis_count=$((network_analysis_count + 1))
        else
            failed_network="${failed_network} ${scenario_num}"
        fi

        sleep 1
    done

    print_success "Network Analysis Phase Complete: ${network_analysis_count}/${trained_count} succeeded"

    # Phase 3: Behavior Analysis
    print_header "PHASE 3/3: BEHAVIOR ANALYSIS"

    for scenario_num in 1 2 3 4 5 6 7 8 9 10 11; do
        # Skip if training failed
        if echo "$failed_training" | grep -q " ${scenario_num}"; then
            print_warning "Skipping behavior analysis for Scenario ${scenario_num} (training failed)"
            continue
        fi

        config_path=$(get_config_path $scenario_num)
        scenario_name=$(get_scenario_name $scenario_num)

        if analyze_behavior "${scenario_num}" "${scenario_name}" "${config_path}"; then
            behavior_analysis_count=$((behavior_analysis_count + 1))
        else
            failed_behavior="${failed_behavior} ${scenario_num}"
        fi

        sleep 1
    done

    print_success "Behavior Analysis Phase Complete: ${behavior_analysis_count}/${trained_count} succeeded"

    # Final Summary
    print_header "PIPELINE COMPLETE - FINAL SUMMARY"

    echo -e "${GREEN}Training:${NC}          ${trained_count}/${total_scenarios} scenarios succeeded"
    echo -e "${GREEN}Network Analysis:${NC}  ${network_analysis_count}/${trained_count} succeeded"
    echo -e "${GREEN}Behavior Analysis:${NC} ${behavior_analysis_count}/${trained_count} succeeded"

    if [ -n "$failed_training" ]; then
        echo -e "\n${RED}Failed Training:${NC} Scenarios${failed_training}"
    fi

    if [ -n "$failed_network" ]; then
        echo -e "${RED}Failed Network Analysis:${NC} Scenarios${failed_network}"
    fi

    if [ -n "$failed_behavior" ]; then
        echo -e "${RED}Failed Behavior Analysis:${NC} Scenarios${failed_behavior}"
    fi

    echo -e "\n${BLUE}Results saved in:${NC} ${BASE_RESULTS_DIR}"
    echo -e "${BLUE}Structure:${NC}"
    echo "  - scenario_X_training.log"
    echo "  - scenario_X_model_path.txt"
    echo "  - scenario_X_network_analysis.log"
    echo "  - scenario_X_network_plots/"
    echo "  - scenario_X_behavior_analysis.log"
    echo "  - scenario_X_behavior/"

    # Generate summary report
    local summary_file="${BASE_RESULTS_DIR}/PIPELINE_SUMMARY.txt"
    {
        echo "============================================================================"
        echo "Unified Reward Function - Complete Pipeline Summary"
        echo "============================================================================"
        echo ""
        echo "Timestamp: ${TIMESTAMP}"
        echo "Total Scenarios: ${total_scenarios}"
        echo ""
        echo "Results:"
        echo "  - Training: ${trained_count}/${total_scenarios} succeeded"
        echo "  - Network Analysis: ${network_analysis_count}/${trained_count} succeeded"
        echo "  - Behavior Analysis: ${behavior_analysis_count}/${trained_count} succeeded"
        echo ""
        if [ -n "$failed_training" ]; then
            echo "Failed Training: Scenarios${failed_training}"
        fi
        if [ -n "$failed_network" ]; then
            echo "Failed Network Analysis: Scenarios${failed_network}"
        fi
        if [ -n "$failed_behavior" ]; then
            echo "Failed Behavior Analysis: Scenarios${failed_behavior}"
        fi
        echo ""
        echo "Results Directory: ${BASE_RESULTS_DIR}"
        echo "============================================================================"
    } > "${summary_file}"

    print_success "Summary report saved: ${summary_file}"

    # Check if all succeeded
    if [ ${trained_count} -eq ${total_scenarios} ] && \
       [ ${network_analysis_count} -eq ${trained_count} ] && \
       [ ${behavior_analysis_count} -eq ${trained_count} ]; then
        print_header "🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY! 🎉"
        return 0
    else
        print_header "⚠️  PIPELINE COMPLETED WITH SOME FAILURES"
        return 1
    fi
}

# Run main pipeline
main

# Exit with appropriate code
exit $?
