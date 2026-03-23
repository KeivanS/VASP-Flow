#!/bin/bash
# Test script for VASP Workflow Agent

echo "=========================================="
echo "VASP Workflow Agent - Installation Test"
echo "=========================================="
echo ""

# Check if Python is available
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python found: $PYTHON_VERSION"
else
    echo "✗ Python 3 not found. Please install Python 3.6 or later."
    exit 1
fi

# Check if main script exists
echo ""
echo "Checking main script..."
if [ -f "vasp-agent.py" ]; then
    echo "✓ vasp-agent.py found"
    if [ -x "vasp-agent.py" ]; then
        echo "✓ vasp-agent.py is executable"
    else
        echo "! vasp-agent.py is not executable. Making it executable..."
        chmod +x vasp-agent.py
    fi
else
    echo "✗ vasp-agent.py not found"
    exit 1
fi

# Check modules
echo ""
echo "Checking modules..."
if [ -d "modules" ]; then
    echo "✓ modules/ directory found"
    if [ -f "modules/instruction_parser.py" ]; then
        echo "  ✓ instruction_parser.py found"
    fi
    if [ -f "modules/vasp_input_generator.py" ]; then
        echo "  ✓ vasp_input_generator.py found"
    fi
else
    echo "✗ modules/ directory not found"
    exit 1
fi

# Check examples
echo ""
echo "Checking examples..."
if [ -d "examples" ]; then
    echo "✓ examples/ directory found"
    EXAMPLE_COUNT=$(ls -1 examples/*.txt 2>/dev/null | wc -l)
    echo "  Found $EXAMPLE_COUNT example instruction files"
else
    echo "! examples/ directory not found (optional)"
fi

# Test run with example
echo ""
echo "Running test with example..."
if [ -f "examples/example1_basic.txt" ] && [ -f "examples/POSCAR_MoS2" ]; then
    echo "Executing: ./vasp-agent.py -p test_installation -i examples/example1_basic.txt -s examples/POSCAR_MoS2"
    ./vasp-agent.py -p test_installation -i examples/example1_basic.txt -s examples/POSCAR_MoS2
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✓ Installation test PASSED!"
        echo "=========================================="
        echo ""
        echo "Test project created at: projects/test_installation/"
        echo ""
        echo "Next steps:"
        echo "1. Review the generated files:"
        echo "   cd projects/test_installation/generated"
        echo "   ls -la"
        echo ""
        echo "2. Configure POTCAR paths:"
        echo "   nano 01_relax/make_potcar.sh"
        echo ""
        echo "3. Read the manual:"
        echo "   pdflatex manual.tex"
        echo ""
        echo "4. Try with your own structure!"
        echo ""
    else
        echo ""
        echo "✗ Test run failed. Check error messages above."
        exit 1
    fi
else
    echo "! Example files not found. Cannot run test."
    echo "  But installation appears to be correct."
fi

echo ""
echo "Installation check complete."
