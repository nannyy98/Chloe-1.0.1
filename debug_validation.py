#!/usr/bin/env python3
"""
Debug test for reasoning core validation issues
"""
import asyncio
import json
from core.enhanced_reasoning_core import EnhancedReasoningCore
from utils.config import Config

async def debug_validation():
    print("🔍 Debug Validation Test")
    print("=" * 40)
    
    config = Config()
    core = EnhancedReasoningCore(config)
    
    # Test 1: Direct analysis
    print("\n1️⃣ Testing direct analysis...")
    try:
        analysis = await core._analyze_task('hello', {})
        print("✅ Analysis successful")
        print(f"   Result: {json.dumps(analysis, indent=2, ensure_ascii=False)}")
        
        # Test validation
        is_valid = core._validate_analysis(analysis)
        print(f"   Validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
    
    # Test 2: Full processing
    print("\n2️⃣ Testing full processing...")
    try:
        result = await core.process('hello', {})
        print("✅ Full processing successful")
        print(f"   Provider: {result.get('provider_used', 'unknown')}")
        print(f"   Confidence: {result.get('confidence', 0) * 100:.0f}%")
        print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
        
        if 'error' in result:
            print(f"   Error: {result['error']}")
        else:
            print("   No errors detected")
            
    except Exception as e:
        print(f"❌ Full processing failed: {e}")
    
    # Test 3: Check validation methods
    print("\n3️⃣ Testing validation methods...")
    
    # Test analysis validation
    test_analysis = {
        "task_type": "other",
        "complexity": "simple", 
        "required_approach": "reasoning"
    }
    analysis_valid = core._validate_analysis(test_analysis)
    print(f"   Analysis validation: {'✅' if analysis_valid else '❌'}")
    
    # Test reasoning validation  
    test_reasoning = {
        "understanding": "Basic understanding",
        "approach": "Direct approach",
        "confidence": 0.7
    }
    reasoning_valid = core._validate_reasoning(test_reasoning)
    print(f"   Reasoning validation: {'✅' if reasoning_valid else '❌'}")

if __name__ == "__main__":
    asyncio.run(debug_validation())