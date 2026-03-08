import os
import json
from datetime import datetime
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
from ddgs import DDGS

load_dotenv()

@tool
def search_web(query: str) -> str:
    """
    Пошук інформації в інтернеті
    Args:
        query: Текст запиту для пошуку
    """
    print(f"Пошук інформації по темі: {query}")
    try:
        with DDGS(timeout=10) as ddgs:
            print("requst with timeout")
            results_iter = ddgs.text(
                query,
                max_results=3,
                backend="duckduckgo",
            )
            print("got results for request")

            results = []
            for r in results_iter:
                results.append(r)
                if len(results) >= 3:
                    break

            if not results:
                return "Нічого не знайдено"

            output = "Результати пошуку:\n\n"

            for i, r in enumerate(results, 1):
                output += f"{i}. {r.get('title', '')}\n"
                output += f"   {r.get('body', '')[:200]}...\n"
                output += f"   {r.get('href', '')}\n\n"

            return output
    except Exception as e:
        print(f" Помилка пошуку: {e}")
        return "Помилка пошуку"
    
@tool
def get_current_date() -> str:
    """
    Отримати поточний час
    """
    now = datetime.now()
    print(f"Поточна дата {now}")
    return now.strftime("%Y-%m-%d")


@tool
def save_report(content: dict, filename: str) -> None:
    """
    Збереження файлу звіту на диск
    Args:
        content: dict з маршрутами для збереженя
        filename: Назва файлу для збереження
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)
    
    print(f"\nзвіт: {filename}")


# 6. Запустіть агента
def main():
    """Запуск LangChain 1.0 агента"""

    # считуємо API ключ
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"   API ключ: {api_key[:7]}...")
    else:
        print(f"   API ключ: не знайдено")

    print("✅ Using LangChain 1.0+ create_agent API\n")
    tools = [search_web, get_current_date, save_report ]
    agent = create_agent(
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="""
Ви — розумний помічник для людини яка хоче відпочити в Україні. 

У вас є доступ до наступних інструментів:
- get_current_date щоб дізнатись поточний час
- search_web для пошуку інформації з DuckDuckGo 
- save_report для зберження результатів в json файл

Як досвідчений тур-агент ви маєте:
- Знайти рекомендації щодо можливих активностей відповідно до запиту туриста;
- Узагальнити знайдену інформацію та побудуйте 5 варіантів маршрутів двотижневої відпустки з урахуванням  поточної дати та безпекової ситуації в можливих місцях відпочинку
- Для назви використовуй формат "AI-generated-routes-<Current_time>.txt  де <Current_time> це поточний час отриманий за допогою інструменту get_current_date
- Запропоновані варіанти маршрутів збережіть у json файл формату {"topic": <запиту користувача>, "timestamp": <поточна дата>, "result": <маршрути які ти підготував> }

Використовуй лише інформацію з контексту.

"""
    )
    

    for step in agent.stream({
        "messages": [{"role": "user", "content": "Відпочинок з відвідуванням семи чудес України і семи природних чудес України"}]
    },
    stream_mode="updates"):
        print(step)




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрограму перервано")
    except Exception as e:
        print(f"\nКритична помилка: {e}")
        import traceback
        traceback.print_exc()