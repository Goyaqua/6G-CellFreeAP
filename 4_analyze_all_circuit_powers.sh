#!/bin/bash

################################################################################
# Script 4: Analyze Circuit Power Impact on All Models
#
# Purpose: Test how different circuit power values affect trained models
#          Analyze power consumption breakdown (TX + Circuit)
# Output: results/circuit_power_YYYYMMDD_HHMMSS/
#
# Usage: bash 4_analyze_all_circuit_powers.sh
################################################################################

set -e


# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="./results/circuit_power_${TIMESTAMP}"
EXPERIMENTS_BASE="./experiments"

mkdir -p "${RESULTS_DIR}"

################################################################################
# Scenario Metadata - Bash 3.2 Compatible
################################################################################

get_scenario_users() {
    case $1 in
        1) echo 16 ;;
        2) echo 8 ;;
        3) echo 32 ;;
        4) echo 16 ;;
        5) echo 16 ;;
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

# Circuit power test values (Watts)
CIRCUIT_POWERS=(0.05 0.1 0.15 0.2 0.3 0.4 0.5)

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "\n${MAGENTA}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════════${NC}\n"
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
# Circuit Power Analysis
################################################################################

analyze_circuit_power() {
    local scenario_num=$1
    local model_dir=$2
    local num_users=$(get_scenario_users $scenario_num)
    local scenario_name=$(get_scenario_name $scenario_num)

    print_header "⚡ CIRCUIT POWER ANALYSIS - Scenario ${scenario_num}: ${scenario_name}"

    local model_path="${model_dir}/models/ppo_cellfree_final"
    local output_dir="${RESULTS_DIR}/scenario_${scenario_num}_${scenario_name}"

    if [ ! -f "${model_path}.zip" ]; then
        print_error "Model not found: ${model_path}.zip"
        return 1
    fi

    mkdir -p "${output_dir}"

    print_info "Model: ${model_path}"
    print_info "Network: 64 APs, ${num_users} Users"
    print_info "Testing ${#CIRCUIT_POWERS[@]} circuit power values"

    # Run analysis for each circuit power value
    for circuit_power in "${CIRCUIT_POWERS[@]}"; do
        print_info "  Testing circuit_power = ${circuit_power}W (${circuit_power}00mW)"

        local log_file="${output_dir}/circuit_${circuit_power}W.log"

        if python src/circuit_power_analyze.py \
            --rl-model "${model_path}" \
            --agent-type ppo \
            --num-aps 64 \
            --num-users ${num_users} \
            --circuit-powers ${circuit_power} \
            --episodes 50 \
            > "${log_file}" 2>&1; then

            print_success "  ✓ Circuit power ${circuit_power}W analyzed"
        else
            print_error "  ✗ Failed for circuit power ${circuit_power}W"
        fi
    done

    # Generate comparison plot (if script exists)
    if [ -f "src/plot_circuit_power_comparison.py" ]; then
        print_info "Generating comparison plots..."
        python src/plot_circuit_power_comparison.py \
            --input-dir "${output_dir}" \
            --output "${output_dir}/circuit_power_comparison.png" \
            2>&1 | tee -a "${output_dir}/comparison.log" || true
    fi

    print_success "Circuit power analysis completed for Scenario ${scenario_num}"
    print_success "Results saved to: ${output_dir}"

    return 0
}

################################################################################
# Main Pipeline
################################################################################

main() {
    print_header "⚡ CIRCUIT POWER IMPACT ANALYSIS FOR ALL MODELS"

    print_info "Analysis Timestamp: ${TIMESTAMP}"
    print_info "Results Directory: ${RESULTS_DIR}"
    print_info ""
    print_info "This analysis tests how circuit power affects:"
    print_info "  - Total power consumption (TX + Circuit)"
    print_info "  - Energy efficiency"
    print_info "  - AP activation decisions"
    print_info ""
    print_info "Circuit power values tested (Watts): ${CIRCUIT_POWERS[*]}"

    local total_scenarios=5
    local success_count=0

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

        # Run circuit power analysis
        if analyze_circuit_power "$scenario_num" "$model_dir"; then
            success_count=$((success_count + 1))
        fi

        sleep 1
    done

    # Summary
    print_header "⚡ CIRCUIT POWER ANALYSIS SUMMARY"

    echo -e "${GREEN}Successful:${NC} ${success_count}/${total_scenarios}"

    echo -e "\n${BLUE}Results saved in:${NC} ${RESULTS_DIR}/"
    echo "  Structure per scenario:"
    echo "    - scenario_X_Name/"
    echo "      ├── circuit_0.05W.log"
    echo "      ├── circuit_0.10W.log"
    echo "      ├── ..."
    echo "      └── circuit_power_comparison.png"

    # Generate summary
    {
        echo "Circuit Power Analysis - ${TIMESTAMP}"
        echo "=========================================="
        echo "Total Scenarios: ${total_scenarios}"
        echo "Successful: ${success_count}"
        echo ""
        echo "Circuit Power Values Tested (W): ${CIRCUIT_POWERS[*]}"
        echo ""
        echo "Results Directory: ${RESULTS_DIR}/"
    } > "${RESULTS_DIR}/CIRCUIT_POWER_SUMMARY.txt"

    print_success "Summary saved: ${RESULTS_DIR}/CIRCUIT_POWER_SUMMARY.txt"

    if [ ${success_count} -eq ${total_scenarios} ]; then
        print_header "🎉 ALL CIRCUIT POWER ANALYSES COMPLETED!"
        return 0
    else
        print_header "⚠️  CIRCUIT POWER ANALYSIS COMPLETED WITH SOME FAILURES"
        return 1
    fi
}

# Run main
main

exit $?
