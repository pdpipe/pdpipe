#!/usr/bin/env python3
"""Test script to verify the fix for issue #70"""

import sys
import os

# Try to import pdpipe, fallback to path manipulation if needed
try:
    import pdpipe as pdp
except ImportError:
    # Fallback for development/testing
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    import pdpipe as pdp

import pandas as pd

def test_user_precondition_error():
    """Test that user precondition errors show correct messages"""
    print("Testing user precondition error...")

    # Create test data - note that column 'x' does not exist
    df = pd.DataFrame([[1, 4], [4, 5], [1, 11]], [1, 2, 3], ["a", "b"])
    print("DataFrame columns:", df.columns.tolist())
    
    # Create pipeline with user precondition that will fail
    pline = pdp.PdPipeline(
        [pdp.FreqDrop(2, "a", prec=pdp.cond.HasAllColumns(["x"]))]
    )
    
    try:
        pline.apply(df)
        print("ERROR: Expected FailedPreconditionError was not raised!")
        return False
    except Exception as e:
        print("Exception type:", type(e).__name__)
        print("Exception message:", str(e))
        
        # Check if this is the right type of error for user precondition
        if (
            "column x" in str(e).lower()
            or "user-provided precondition" in str(e).lower()
        ):
            print("✓ SUCCESS: User precondition error message is correct!")
            return True
        else:
            print(
                "✗ FAILURE: Error message does not indicate user precondition failure"
            )
            print(
                "Expected: Message about column 'x' not found (user precondition)"
            )
            print("Got:", str(e))
            return False

def test_stage_precondition_error():
    """Test that stage precondition errors still work correctly"""
    print("\nTesting stage precondition error...")

    # Create test data without column 'a' to trigger stage precondition failure
    df = pd.DataFrame([[1, 4], [4, 5], [1, 11]], [1, 2, 3], ["x", "y"])
    print("DataFrame columns:", df.columns.tolist())

    # Create pipeline without user precondition - should trigger stage precondition
    pline = pdp.PdPipeline([pdp.FreqDrop(2, "a")])  # column 'a' doesn't exist
    
    try:
        pline.apply(df)
        print("ERROR: Expected FailedPreconditionError was not raised!")
        return False
    except Exception as e:
        print("Exception type:", type(e).__name__)
        print("Exception message:", str(e))
        
        # Check if this shows stage precondition error
        if "column a" in str(e).lower() and "freqdrop" in str(e).lower():
            print("✓ SUCCESS: Stage precondition error message is correct!")
            return True
        else:
            print("✗ FAILURE: Stage precondition error message is not correct")
            return False

def test_successful_case():
    """Test that normal operation still works"""
    print("\nTesting successful case...")

    # Create test data where everything should work
    df = pd.DataFrame([[1, 4], [4, 5], [1, 11]], [1, 2, 3], ["a", "b"])
    print("DataFrame columns:", df.columns.tolist())

    # Create pipeline with user precondition that should pass
    pline = pdp.PdPipeline(
        [pdp.FreqDrop(2, "a", prec=pdp.cond.HasAllColumns(["a", "b"]))]
    )
    
    try:
        result = pline.apply(df)
        print("✓ SUCCESS: Pipeline executed successfully!")
        print("Result shape:", result.shape)
        return True
    except Exception as e:
        print("✗ FAILURE: Unexpected error:", str(e))
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing fix for issue #70: Incorrect precondition error messages")
    print("=" * 60)

    test1 = test_user_precondition_error()
    test2 = test_stage_precondition_error()
    test3 = test_successful_case()

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"User precondition test: {'PASS' if test1 else 'FAIL'}")
    print(f"Stage precondition test: {'PASS' if test2 else 'FAIL'}")
    print(f"Successful case test: {'PASS' if test3 else 'FAIL'}")
    print("=" * 60)

    if all([test1, test2, test3]):
        print("✓ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED!")
        sys.exit(1)