#!/bin/bash

################################################################################
# Script 2: Analyze All Trained Models (Performance & Behavior)
#
# Purpose: Run performance and behavior analysis on all trained models
# Output: results/analysis_YYYYMMDD_HHMMSS/
#
# Usage: bash 2_analyze_all_models.sh [training_timestamp]
#        or bash 2_analyze_all_models.sh  (will auto-find latest)
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="./results/analysis_${TIMESTAMP}"
EXPERIMENTS_BASE="./experiments"

mkdir -p "${RESULTS_DIR}"

################################################################################
# Scenario Metadata (64 APs) - Bash 3.2 Compatible
################################################################################

get_scenario_users() {
    case $1 in
        1) echo 16 ;;  # Balanced
        2) echo 8 ;;   # Low Density
        3) echo 32 ;;  # High Density
        4) echo 16 ;;  # QoS Priority
        5) echo 16 ;;  # Green Network
    esac
}

get_scenario_name() {
    case $1 in
        1) echo "Balanced_Baseline" ;;
        2) echo "Low_Density_Shutdown" ;;
        3) echo "High_Density_Interference" ;;
        4) echo "QoS_Priority" ;;
        5) echo "Green_Network" ;;
    esac
}

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}\n"
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${YELLOW}ℹ️  $1${NC}"; }

################################################################################
# Find Models
################################################################################

find_latest_training() {
    local pattern=$1
    local latest=$(ls -td ${EXPERIMENTS_BASE}/*_scenario_${pattern}* 2>/dev/null | head -1)
    echo "$latest"
}

################################################################################
# Performance Analysis
################################################################################

analyze_performance() {
    local scenario_num=$1
    local model_dir=$2
    local scenario_name=$(get_scenario_name $scenario_num)

    print_header "PERFORMANCE ANALYSIS - Scenario ${scenario_num}: ${scenario_name}"

    local model_path="${model_dir}/models/ppo_cellfree_final"

    # Find config file using wildcard pattern (same as analyze_behavior)
    local config_path="configs/ppo_scenarios/${scenario_num}_*.yaml"
    config_path=$(ls ${config_path} 2>/dev/null | head -1)

    local output_dir="${RESULTS_DIR}/scenario_${scenario_num}_performance"

    if [ ! -f "${model_path}.zip" ]; then
        print_error "Model not found: ${model_path}.zip"
        return 1
    fi

    if [ -z "${config_path}" ] || [ ! -f "${config_path}" ]; then
        print_error "Config not found: configs/ppo_scenarios/${scenario_num}_*.yaml"
        return 1
    fi

    mkdir -p "${output_dir}"

    print_info "Model: ${model_path}"
    print_info "Config: ${config_path}"
    print_info "Output: ${output_dir}"

    # FIXED: Call analyze_network.py with correct arguments
    if python src/analyze_network.py \
        --model "${model_path}" \
        --config "${config_path}" \
        --episodes 100 \
        > "${output_dir}/performance.log" 2>&1; then

        print_success "Performance analysis completed"

        # Move generated plots from default location to output directory
        if [ -d "results/evaluate" ]; then
            print_info "Moving plots to ${output_dir}/"
            mv results/evaluate/* "${output_dir}/" 2>/dev/null || true
            rmdir results/evaluate 2>/dev/null || true
        fi

        return 0
    else
        print_error "Performance analysis failed"
        print_error "Check log: ${output_dir}/performance.log"
        return 1
    fi
}

################################################################################
# Behavior Analysis
################################################################################

analyze_behavior() {
    local scenario_num=$1
    local model_dir=$2
    local scenario_name=$(get_scenario_name $scenario_num)

    print_header "BEHAVIOR ANALYSIS - Scenario ${scenario_num}: ${scenario_name}"

    local model_path="${model_dir}/models/ppo_cellfree_final"
    local config_path="configs/ppo_scenarios/${scenario_num}_*.yaml"
    config_path=$(ls ${config_path} 2>/dev/null | head -1)
    local output_dir="${RESULTS_DIR}/scenario_${scenario_num}_behavior"
    local log_file="${RESULTS_DIR}/scenario_${scenario_num}_behavior.log"

    if [ ! -f "${model_path}.zip" ]; then
        print_error "Model not found: ${model_path}.zip"
        return 1
    fi

    if [ ! -f "${config_path}" ]; then
        print_error "Config not found for scenario ${scenario_num}"
        return 1
    fi

    mkdir -p "${output_dir}"

    print_info "Model: ${model_path}"
    print_info "Config: ${config_path}"
    print_info "Output: ${output_dir}"

    if python src/analyze_behavior.py \
        --model "${model_path}" \
        --config "${config_path}" \
        --episodes 100 \
        --save_dir "${output_dir}" \
        > "${log_file}" 2>&1; then

        print_success "Behavior analysis completed"
        print_success "Results saved to: ${output_dir}"
        return 0
    else
        print_error "Behavior analysis failed"
        print_error "Check log: ${log_file}"
        return 1
    fi
}

################################################################################
# Main Pipeline
################################################################################

main() {
    local training_timestamp=$1

    print_header "📊 ANALYZING ALL TRAINED MODELS"

    print_info "Analysis Timestamp: ${TIMESTAMP}"
    print_info "Results Directory: ${RESULTS_DIR}"

    local total_scenarios=5
    local performance_success=0
    local behavior_success=0

    # Analyze each scenario
    for scenario_num in 1 2 3 4 5; do
        local scenario_name=$(get_scenario_name $scenario_num)

        # Find model directory
        local model_dir=$(find_latest_training "${scenario_num}_${scenario_name}")

        if [ -z "$model_dir" ]; then
            print_warning "No model found for Scenario ${scenario_num}"
            continue
        fi

        print_info "Found model: ${model_dir}"

        # Run performance analysis
        if analyze_performance "$scenario_num" "$model_dir"; then
            performance_success=$((performance_success + 1))
        fi

        # Run behavior analysis
        if analyze_behavior "$scenario_num" "$model_dir"; then
            behavior_success=$((behavior_success + 1))
        fi

        sleep 1
    done

    # Summary
    print_header "📊 ANALYSIS SUMMARY"

    echo -e "${GREEN}Performance Analysis:${NC} ${performance_success}/${total_scenarios} succeeded"
    echo -e "${GREEN}Behavior Analysis:${NC} ${behavior_success}/${total_scenarios} succeeded"

    echo -e "\n${BLUE}Results saved in:${NC} ${RESULTS_DIR}/"
    echo "  Structure:"
    echo "    - scenario_X_performance/"
    echo "    - scenario_X_performance.log"
    echo "    - scenario_X_behavior/"
    echo "    - scenario_X_behavior.log"

    # Generate summary
    {
        echo "Model Analysis Summary - ${TIMESTAMP}"
        echo "==========================================="
        echo "Total Scenarios: ${total_scenarios}"
        echo "Performance Analysis Success: ${performance_success}"
        echo "Behavior Analysis Success: ${behavior_success}"
        echo ""
        echo "Results Directory: ${RESULTS_DIR}/"
    } > "${RESULTS_DIR}/ANALYSIS_SUMMARY.txt"

    print_success "Summary saved: ${RESULTS_DIR}/ANALYSIS_SUMMARY.txt"

    if [ ${performance_success} -eq ${total_scenarios} ] && [ ${behavior_success} -eq ${total_scenarios} ]; then
        print_header "🎉 ALL ANALYSES COMPLETED SUCCESSFULLY!"
        return 0
    else
        print_header "⚠️  ANALYSIS COMPLETED WITH SOME FAILURES"
        return 1
    fi
}

# Run main
main "$@"

exit $?
