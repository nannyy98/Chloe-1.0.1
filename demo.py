#!/usr/bin/env python3
"""
Demo Script - Демонстрация работы Chloe AI
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import Config
from utils.logger import setup_logger
from core.enhanced_reasoning_core import EnhancedReasoningCore
from agents.tool_manager import ToolManager
from memory.memory_system import MemorySystem
from learning.learning_engine import LearningEngine

async def demo_chloe_ai():
    """Демонстрация работы системы"""
    print("🤖 Chloe AI - Демонстрация работы")
    print("=" * 50)
    
    # Инициализация
    config = Config()
    logger = setup_logger("demo")
    
    print("1️⃣  Инициализация компонентов...")
    reasoning_core = EnhancedReasoningCore(config)
    tool_manager = ToolManager(config)
    memory_system = MemorySystem(config)
    learning_engine = LearningEngine(config, memory_system)
    
    print("   ✓ Все компоненты загружены")
    
    # Демонстрация 1: Решение задачи через reasoning
    print("\n2️⃣  Демонстрация reasoning...")
    task = "Объясни что такое машинное обучение простыми словами"
    
    try:
        result = await reasoning_core.process(task)
        print(f"   Задача: {task}")
        print(f"   Результат: {result.get('reasoning', {}).get('understanding', 'Нет результата')}")
        print(f"   Уверенность: {result.get('confidence', 0):.2f}")
        
        # Сохраняем в память
        await memory_system.store_interaction(task, result)
        await learning_engine.record_experience(task, {"action": "reason", "confidence": result.get('confidence', 0)}, result)
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    # Демонстрация 2: Использование инструментов
    print("\n3️⃣  Демонстрация инструментов...")
    
    # Список доступных инструментов
    tools = tool_manager.list_available_tools()
    print(f"   Доступные инструменты: {', '.join(tools)}")
    
    # Демонстрация code agent
    print("\n   🔧 Code Agent demo:")
    code_task = {
        "task_type": "run_code",
        "code": "print('Hello from Chloe AI!')\nprint(2 + 2)"
    }
    
    try:
        code_result = await tool_manager.execute_tool("code_runner", code_task)
        print(f"   Результат выполнения кода: {code_result}")
    except Exception as e:
        print(f"   ❌ Ошибка выполнения кода: {e}")
    
    # Демонстрация file agent
    print("\n   📁 File Agent demo:")
    file_task = {
        "task_type": "write",
        "path": "demo_file.txt",
        "content": "Это демонстрационный файл, созданный Chloe AI"
    }
    
    try:
        file_result = await tool_manager.execute_tool("file_agent", file_task)
        print(f"   Результат работы с файлами: {file_result}")
    except Exception as e:
        print(f"   ❌ Ошибка работы с файлами: {e}")
    
    # Демонстрация 3: Память и обучение
    print("\n4️⃣  Демонстрация памяти и обучения...")
    
    try:
        # Получаем статистику памяти
        memory_stats = await memory_system.get_memory_stats()
        print(f"   Статистика памяти: {memory_stats}")
        
        # Получаем состояние обучения
        learning_state = await learning_engine.get_current_state()
        print(f"   Состояние обучения: {learning_state['recent_success_rate']:.2%} успешных выполнений")
        
    except Exception as e:
        print(f"   ❌ Ошибка работы с памятью/обучением: {e}")
    
    # Демонстрация 4: Комплексная задача
    print("\n5️⃣  Комплексная задача...")
    complex_task = "Напиши Python функцию для расчета факториала и протестируй ее"
    
    try:
        # Сначала генерируем код через reasoning
        code_generation = await reasoning_core.generate_code(
            "Python функция для расчета факториала с тестами",
            {"task": complex_task}
        )
        
        print(f"   Сгенерированный код:\n{code_generation}")
        
        # Выполняем код
        execution_task = {
            "task_type": "run_code",
            "code": code_generation
        }
        
        execution_result = await tool_manager.execute_tool("code_runner", execution_task)
        print(f"   Результат выполнения:\n{execution_result}")
        
        # Сохраняем опыт
        await memory_system.store_interaction(complex_task, execution_result)
        await learning_engine.record_experience(
            complex_task, 
            {"action": "tool", "confidence": 0.9}, 
            execution_result
        )
        
    except Exception as e:
        print(f"   ❌ Ошибка комплексной задачи: {e}")
    
    # Финальная статистика
    print("\n" + "=" * 50)
    print("📊 ФИНАЛЬНАЯ СТАТИСТИКА")
    print("=" * 50)
    
    try:
        final_stats = await memory_system.get_memory_stats()
        learning_metrics = learning_engine.get_learning_metrics()
        
        print(f"🧠 Всего взаимодействий: {final_stats['short_term_interactions']}")
        print(f"📚 Знаний в базе: {final_stats['knowledge_base']}")
        print(f"📈 Опыт обучения: {learning_metrics['experience_count']}")
        print(f"🎯 Уровень успеха: {learning_metrics['recent_performance']:.2%}")
        
    except Exception as e:
        print(f"❌ Ошибка получения статистики: {e}")
    
    print("\n🎉 Демонстрация завершена!")
    print("Система готова к реальному использованию!")

if __name__ == "__main__":
    print("Запуск демонстрации Chloe AI...")
    try:
        asyncio.run(demo_chloe_ai())
    except KeyboardInterrupt:
        print("\n\nДемонстрация прервана пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()