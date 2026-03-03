import os
import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from ddgs import DDGS

load_dotenv()


def search_web(query: str) -> str:
    """Пошук інформації в інтернеті"""
    print(f"Пошук інформації по темі: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                output = "Результати пошуку:\n\n"
                for i, r in enumerate(results, 1):
                    output += f"{i}. {r.get('title', '')}\n"
                    output += f"   {r.get('body', '')[:200]}...\n"
                    output += f"   {r.get('href', '')}\n\n"
                return output
    except Exception as e:
        print(f" Помилка пошуку: {e}")
        return "Помилка пошуку"
    

def get_current_date() -> datetime:
    today = datetime.today()
    print(f"Поточна дата {today}")
    return today


# 4. Створіть функцію збереження звіту
def save_report(content: dict, filename: str) -> None:
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
    
    # 1. Створіть LLM
    llm = ChatOpenAI(
        model="gpt-5",
        temperature=0.7,
        api_key=api_key
    )
    print(f" ChatOpenAI LLM створено")


    # Пошук вхідних данних
    topic1 = "Сім чудес України"
    topic2 = "Сім природних чудес України"
 
    search_results1 = search_web(query=topic1)
    search_results2 = search_web(query=topic2)

    # 5. Створіть ланцюжок (chain) prompt → llm → parser
    system_prompt = f"Ви — розумний помічник для людини яка хоче відпочити в Україні. Як досвідчений тур-агент ви маєте дати рекомендації щодо наданого списку місь та можливих активностей. Використовуй лише інформацію з контексту."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Тема: {topic}\n\nДані:\n{search_results}\n\nСтворіть детальний аналіз того чим можна зайнятись.")
    ])

    chain = prompt | llm | StrOutputParser()
    
    # Ланцюги Дослідження
    print(topic1)
    city_result = chain.invoke({
                    "topic": topic1,
                    "search_results": search_results1
                })
    print("\n" + "=" * 60)

    print(topic2)
    nature_result = chain.invoke({
                    "topic": topic2,
                    "search_results": search_results2
                })
    print("\n" + "=" * 60)

    # Ланцюг для висновків
    today_date = get_current_date()
    conclusion_prompt = ChatPromptTemplate.from_messages([
        ("system", "Ви експерт з підготовки маршрутів. Узагальніть  надану інформацію та побудуйте 5 варіантів маршрутів двотижневої відпустки з урахуванням наданої дати та поточної безпокової ситуації в можливих місцях відпочинку"),
        ("human", "Можливий відпочинок на природі: {nature}, Можливий відпочинок у містах: {city}. Поточна дата: {date}")
    ])
            
    chain2 = conclusion_prompt | llm | StrOutputParser()
    print("Підготовка висновків")
    conclusion_result = chain2.invoke({
                    "city": city_result,
                    "nature": nature_result,
                    "date": today_date
                })
    print("\n" + "=" * 60)
    
    result = {
        "topic": "Туристичні маршрути Ураїною", 
        "result": conclusion_result,
        "timestamp": today_date.isoformat()

    }
    # Виведення звіту
    print("\n" + "=" * 60)
    print(result)
    save_report(content=result, filename=f"{today_date}.json")
    print("\nГотово! Перегляньте файл:")
    print(f"   - {today_date}.json - повні дані")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрограму перервано")
    except Exception as e:
        print(f"\nКритична помилка: {e}")
        import traceback
        traceback.print_exc()