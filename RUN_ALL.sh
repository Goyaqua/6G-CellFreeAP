#!/bin/bash

################################################################################
# MASTER SCRIPT: Run Complete Experiment Pipeline
#
# This script runs all 4 analysis steps in sequence:
#   1. Train all scenarios
#   2. Analyze all models (performance & behavior)
#   3. Generate star topology heatmaps
#   4. Analyze circuit power impact
#
# Usage: bash RUN_ALL.sh
#        or bash RUN_ALL.sh --skip-training  (if models already trained)
################################################################################

set -e


# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

################################################################################
# Configuration
################################################################################

SKIP_TRAINING=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash RUN_ALL.sh [--skip-training]"
            exit 1
            ;;
    esac
done

################################################################################
# Helper Functions
################################################################################

print_banner() {
    echo -e "\n${BOLD}${MAGENTA}"
    echo "╔═══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                       ║"
    printf "║  %-67s  ║\n" "$1"
    echo "║                                                                       ║"
    echo "╚═══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}\n"
}

print_step() {
    echo -e "\n${CYAN}${BOLD}▶ STEP $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${YELLOW}ℹ️  $1${NC}"; }

run_script() {
    local script=$1
    local description=$2

    if [ ! -f "$script" ]; then
        print_error "Script not found: $script"
        return 1
    fi

    print_info "Running: $description"
    print_info "Script: $script"

    if bash "$script"; then
        print_success "$description completed successfully"
        return 0
    else
        print_error "$description failed!"
        return 1
    fi
}

################################################################################
# Main Pipeline
################################################################################

main() {
    print_banner "🚀 COMPLETE EXPERIMENT PIPELINE"

    echo -e "${BOLD}Configuration:${NC}"
    echo -e "  Skip Training: ${YELLOW}${SKIP_TRAINING}${NC}"
    echo -e "  Total Steps: ${YELLOW}$([ "$SKIP_TRAINING" = true ] && echo "3" || echo "4")${NC}"
    echo ""

    local start_time=$(date +%s)
    local failed_steps=""

    # Step 1: Training (optional)
    if [ "$SKIP_TRAINING" = false ]; then
        print_step "1/4: Training All Scenarios"
        if run_script "1_train_all_scenarios.sh" "Training All Scenarios"; then
            :
        else
            failed_steps="${failed_steps} Training"
            print_warning "Continuing despite training failures..."
        fi
    else
        print_warning "Skipping training (--skip-training flag)"
    fi

    # Step 2: Model Analysis
    local step_num=2
    if [ "$SKIP_TRAINING" = true ]; then
        step_num=1
    fi

    print_step "${step_num}/$([ "$SKIP_TRAINING" = true ] && echo "3" || echo "4"): Analyzing All Models"
    if run_script "2_analyze_all_models.sh" "Model Analysis"; then
        :
    else
        failed_steps="${failed_steps} Analysis"
    fi

    # Step 3: Star Topology Heatmaps
    step_num=$((step_num + 1))

    print_step "${step_num}/$([ "$SKIP_TRAINING" = true ] && echo "3" || echo "4"): Star Topology Heatmaps"
    if run_script "3_analyze_all_star_heatmaps.sh" "Star Topology Analysis"; then
        :
    else
        failed_steps="${failed_steps} StarHeatmap"
    fi

    # Step 4: Circuit Power Analysis
    step_num=$((step_num + 1))

    print_step "${step_num}/$([ "$SKIP_TRAINING" = true ] && echo "3" || echo "4"): Circuit Power Analysis"
    if run_script "4_analyze_all_circuit_powers.sh" "Circuit Power Analysis"; then
        :
    else
        failed_steps="${failed_steps} CircuitPower"
    fi

    # Final Summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))

    print_banner "📊 PIPELINE COMPLETE"

    echo -e "${BOLD}Execution Time:${NC} ${hours}h ${minutes}m ${seconds}s"
    echo ""

    if [ -z "$failed_steps" ]; then
        echo -e "${GREEN}${BOLD}🎉 ALL STEPS COMPLETED SUCCESSFULLY! 🎉${NC}"
        echo ""
        echo -e "${BLUE}Results locations:${NC}"
        echo "  📁 experiments/          - Trained models"
        echo "  📁 logs/                 - Training logs"
        echo "  📁 results/analysis_*    - Performance & behavior"
        echo "  📁 results/star_heatmaps_* - Star topology analysis"
        echo "  📁 results/circuit_power_* - Circuit power analysis"
        return 0
    else
        echo -e "${YELLOW}${BOLD}⚠️  PIPELINE COMPLETED WITH FAILURES${NC}"
        echo -e "${RED}Failed steps:${failed_steps}${NC}"
        echo ""
        echo "Check individual script logs for details"
        return 1
    fi
}

# Run main
main

exit $?
