#!/bin/bash

################################################################################
# Script 3: Analyze All Models on Star Topology (Cross Pattern Heatmaps)
#
# Purpose: Test trained models on star (X-shaped) user distribution
#          Generate heatmaps showing AP activation patterns
# Output: results/star_heatmaps_YYYYMMDD_HHMMSS/
#
# Usage: bash 3_analyze_all_star_heatmaps.sh
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
RESULTS_DIR="./results/star_heatmaps_${TIMESTAMP}"
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
# Star Topology Heatmap Analysis
################################################################################

analyze_star_heatmap() {
    local scenario_num=$1
    local model_dir=$2
    local num_users=$(get_scenario_users $scenario_num)
    local scenario_name=$(get_scenario_name $scenario_num)

    print_header "🌟 STAR HEATMAP - Scenario ${scenario_num}: ${scenario_name}"

    local model_path="${model_dir}/models/ppo_cellfree_final.zip"

    if [ ! -f "${model_path}" ]; then
        print_error "Model not found: ${model_path}"
        return 1
    fi

    print_info "Model: ${model_path}"
    print_info "Network: 64 APs, ${num_users} Users (Star Topology)"
    print_info "Output: ${RESULTS_DIR}"

    # Run star topology analysis using run_star_analysis.py
    if python src/run_star_analysis.py \
        --model "${model_path}" \
        --num-aps 64 \
        --num-users ${num_users} \
        --num-episodes 10 \
        --episode-length 100 \
        --output "${RESULTS_DIR}" \
        2>&1 | tee "${RESULTS_DIR}/scenario_${scenario_num}_star.log"; then

        print_success "Star heatmap analysis completed"
        return 0
    else
        print_error "Star heatmap analysis failed"
        print_error "Check log: ${RESULTS_DIR}/scenario_${scenario_num}_star.log"
        return 1
    fi
}

################################################################################
# Main Pipeline
################################################################################

main() {
    print_header "🌟 STAR TOPOLOGY HEATMAP ANALYSIS FOR ALL MODELS"

    print_info "Analysis Timestamp: ${TIMESTAMP}"
    print_info "Results Directory: ${RESULTS_DIR}"
    print_info ""
    print_info "This analysis tests if RL agents learned the traffic pattern:"
    print_info "  - Users distributed in X-shaped (cross) pattern"
    print_info "  - 64 APs in grid (fixed infrastructure)"
    print_info "  - Agent should activate APs near the cross, turn off distant ones"

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

        # Run star heatmap analysis
        if analyze_star_heatmap "$scenario_num" "$model_dir"; then
            success_count=$((success_count + 1))
        fi

        sleep 1
    done

    # Summary
    print_header "🌟 STAR HEATMAP ANALYSIS SUMMARY"

    echo -e "${GREEN}Successful:${NC} ${success_count}/${total_scenarios}"

    echo -e "\n${BLUE}Heatmaps saved in:${NC} ${RESULTS_DIR}/"
    echo "  Generated files:"
    echo "    - {model_name}_star_topology_heatmap.png"
    echo "    - {model_name}_star_topology_analysis.json"
    echo "    - scenario_X_star.log"

    # Generate summary
    {
        echo "Star Topology Heatmap Analysis - ${TIMESTAMP}"
        echo "================================================"
        echo "Total Scenarios: ${total_scenarios}"
        echo "Successful: ${success_count}"
        echo ""
        echo "Purpose: Test if RL agent learned traffic patterns"
        echo "Topology: 4-avenue cross (X) pattern"
        echo ""
        echo "Results Directory: ${RESULTS_DIR}/"
    } > "${RESULTS_DIR}/STAR_ANALYSIS_SUMMARY.txt"

    print_success "Summary saved: ${RESULTS_DIR}/STAR_ANALYSIS_SUMMARY.txt"

    if [ ${success_count} -eq ${total_scenarios} ]; then
        print_header "🎉 ALL STAR HEATMAPS GENERATED SUCCESSFULLY!"
        return 0
    else
        print_header "⚠️  STAR ANALYSIS COMPLETED WITH SOME FAILURES"
        return 1
    fi
}

# Run main
main

exit $?
