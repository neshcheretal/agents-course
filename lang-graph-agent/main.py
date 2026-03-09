import os
import json
from datetime import datetime
from dotenv import load_dotenv
from ddgs import DDGS
import operator
from typing import TypedDict, Annotated, List
from functools import partial


# LangChain / LangGraph imports
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    print("[OK] LangChain 1.0 компоненти завантажено")
except ImportError as e:
    print(f"[ERROR] Помилка імпорту LangChain: {e}")
    print("Встановіть: pip install langchain-openai langchain-core")
    exit(1)

try:
    from langgraph.graph import StateGraph, END
    print("[OK] LangGraph завантажено")
except ImportError as e:
    print(f"[ERROR] Помилка імпорту LangGraph: {e}")
    print("Встановіть: pip install langgraph")
    exit(1)


load_dotenv()


class AgentState(TypedDict):
    topic: str                                          
    research_results: str
    research_attempt: int
    analysis_results: str
    analysis_base: int
    final_report: str
    final_execution_status: str
    messages: Annotated[List[str], operator.add]
    timestamp: str


def researcher_node(state: AgentState, llm: ChatOpenAI, ) -> AgentState:
    """Агент-дослідник: шукає інформацію"""
    print("RESEARCHER AGENT: Пошук інформації...")

    topic = state["topic"]
    attempt_count = state["research_attempt"]
    current_attempt = attempt_count + 1
    if current_attempt >= 4:
        print("Перевищенно кількість пошукових спроб")
        return {
            "research_results": "Перевищенно кількість пошукових спроб",
            "messages": ["[ERROR] Researcher: Пошук не дає достатніх результатів"],
            "research_attempt":  current_attempt,
            "final_execution_status": "ERROR"
        }

    # Виконуємо пошук
    print(f"Виконуємо спробу пошуку {current_attempt}")
    web_search_results = search_web(topic)
    print(f"\n{web_search_results[:300]}...")

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Головний Дослідник AI.
            Досвід 15 років у дослідженнях
            Ваша експертиза полягає в глибокому зануренні в теми, пошуку достовірних джерел
            Проаналізуйте знайдену інформацію та виділіть 5 ключових фактів."""),
            ("human", "Тема: {topic}\n\nДані:\n{data}")
        ])

        chain = prompt | llm | StrOutputParser()
        ai_summary = chain.invoke({"topic": topic, "data": web_search_results})
        search_results = f"{web_search_results}\n\nAI Висновки:\n{ai_summary}"
    except Exception as e:
        print(f"[WARNING] AI обробка недоступна: {e}")

    return {
        "research_results": search_results,
        "messages": ["[OK] Researcher: Пошук завершено"],
        "research_attempt":  current_attempt,
        "final_execution_status": "OK"
    }


def analyst_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """Агент-аналітик: аналізує дані"""
    print("\n" + "="*60)
    print("ANALYST AGENT: Аналіз даних...")
    print("="*60)

    research_results = state["research_results"]

    # Виконуємо аналіз
    analysis, word_count = analyze_data(research_results)
    print(f"\n{analysis}")

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Ви - експерт з data science та аналізу трендів.
            Проаналізуйте дані та виявіть ключові інсайти, тренди та закономірності."""),
            ("human", "Дані для аналізу:\n{data}")
        ])

        chain = prompt | llm | StrOutputParser()
        deep_analysis = chain.invoke({"data": research_results})
        analysis = f"{analysis}\n\nГлибокий аналіз:\n{deep_analysis}"
    except Exception as e:
        print(f"[WARNING] AI аналіз недоступний: {e}")


    return {
        "analysis_results": analysis,
        "analysis_base": word_count,
        "messages": ["[OK] Analyst: Аналіз завершено"],
        "final_execution_status": "OK"
    }

def reporter_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """Агент-репортер: створює фінальний звіт"""
    print("\n" + "="*60)
    print("REPORTER AGENT: Створення звіту...")
    print("="*60)

    topic = state["topic"]
    research_results = state["research_results"]
    analysis_results = state["analysis_results"]

    # Створюємо базовий звіт
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║         LANGGRAPH MULTI-AGENT RESEARCH REPORT                ║
╚══════════════════════════════════════════════════════════════╝

Дата: {datetime.now():%Y-%m-%d %H:%M:%S}
Тема: {topic}
Платформа: LangChain 1.0 + LangGraph

════════════════════════════════════════════════════════════════
РЕЗУЛЬТАТИ ДОСЛІДЖЕННЯ (Researcher Agent)
════════════════════════════════════════════════════════════════

{research_results}

════════════════════════════════════════════════════════════════
АНАЛІТИКА (Analyst Agent)
════════════════════════════════════════════════════════════════

{analysis_results}
"""

    # Якщо є LLM - додаємо професійні висновки

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Ви - професійний технічний письменник з 10 роками досвіду технічного писання
            Ваш стиль письма чіткий, лаконічний та підкріплений ґрунтовними дослідженнями
            Створіть executive summary та рекомендації на основі дослідження та аналізу."""),
            ("human", "Дослідження:\n{research}\n\nАналіз:\n{analysis}")
        ])

        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"research": research_results, "analysis": analysis_results})
        report += f"""
════════════════════════════════════════════════════════════════
EXECUTIVE SUMMARY (Reporter Agent)
════════════════════════════════════════════════════════════════

{summary}
"""
    except Exception as e:
        print(f"[WARNING] AI генерація висновків недоступна: {e}")

    # Зберігаємо звіт
    save_status = save_report(report, f"langgraph_report_{datetime.now():%Y%m%d_%H%M%S}.md")
    print(f"\n{save_status}")

    return {
        "final_report": report,
        "messages": ["[OK] Reporter: Звіт створено та збережено"],
        "final_execution_status": "OK"
    }

def search_web(query: str) -> str:
    """
    Пошук інформації в інтернеті
    Args:
        query: Текст запиту для пошуку
    """
    print(f"Пошук інформації по темі: {query}")
    try:
        with DDGS(timeout=10) as ddgs:
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

            for i, r in enumerate(results, 0):
                output += f"{i}. {r.get('title', '')}\n"
                output += f"   {r.get('body', '')[:200]}...\n"
                output += f"   {r.get('href', '')}\n\n"

            return output
    except Exception as e:
        print(f" Помилка пошуку: {e}")
        return "Помилка пошуку"



def save_report(content: str, filename: str) -> None:
    """
    Збереження файлу звіту на диск
    Args:
        content: dict з маршрутами для збереженя
        filename: Назва файлу для збереження
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\nзвіт: {filename}")


def analyze_data(text: str) -> tuple:
    """
    Аналіз даних
    Args:
        text: Текст для аналізу
    """
    word_count = len(text.split())
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    # Простий sentiment аналіз
    positive_ai_keywords = [
        "прогрес",
        "інновація",
        "інновації",
        "розвиток",
        "покращення",
        "вдосконалення",
        "оптимізація",
        "модернізація",
        "трансформація",
        "прорив",
        "ефективність",
        "продуктивність",
        "автоматизація",
        "зростання",
        "масштабованість",
        "конкурентоспроможність",
        "прибутковість",
        "створення цінності",
        "підвищення продуктивності",
        "спрощення процесів",
        "підвищення ефективності",
        "висока продуктивність",
        "розумні системи",
        "точність",
        "інтелектуальний",
        "адаптивний",
        "автономний",
        "надійний",
        "стійкий",
        "масштабований",
        "пояснюваний ШІ",
        "надійний ШІ",
        "передові моделі",
        "революційний",
        "трансформаційний",
        "проривна технологія",
        "передовий",
        "інноваційний",
        "технологічний прогрес"
    ]

    negative_ai_keywords = [
        "ризик",
        "загроза",
        "небезпека",
        "невизначеність",
        "нестабільність",
        "вразливість",
        "зловживання",
        "маніпуляція",
        "упередженість",
        "галюцинації ШІ",
        "ризик безпеки",
        "збій системи",
        "втрата роботи",
        "витіснення працівників",
        "ризик автоматизації",
        "безробіття",
        "порушення ринку праці",
        "застарівання навичок",
        "заміна працівників",
        "нерівність",
        "економічні потрясіння",
        "неетичний",
        "непрозорий",
        "нерегульований",
        "спостереження",
        "порушення приватності",
        "дезінформація",
        "діпфейк",
        "відсутність відповідальності",
        "експлуатація даних",
        "катастрофічний",
        "небезпечний",
        "неконтрольований",
        "шкідливий",
        "екзистенційний ризик"
    ]

    pos_count = sum(1 for word in positive_ai_keywords if word in text.lower())
    neg_count = sum(1 for word in negative_ai_keywords if word in text.lower())
    
    sentiment = "позитивний" if pos_count > neg_count else "негативний" if neg_count > pos_count else "нейтральний"
    
    return f"""
Аналіз даних:
- Слів: {word_count}
- Речень: {sentences}
- Тональність: {sentiment}
- Позитивних маркерів: {pos_count}
- Негативних маркерів: {neg_count}
""", word_count


def route_after_analyst(state: AgentState) -> str:
    """Визначає наступний крок"""
    analysis_base_word_count = state.get("analysis_base", 0)
 
    if analysis_base_word_count <= 100:
        print("Недостатня кількість данних для грунтовного аналізу: Повторити researcher")
        return "researcher"
    else:
        return "reporter"
    
def route_decision_research(state: AgentState) -> str:
    """Визначає наступний крок"""
    research_attempt = state.get("research_attempt", 0)
  
    if research_attempt >=4:
        print("Перевищена допустима кількість спроб пошуку: завершення циклу")
        return "end"
    else:
        return "analyst"


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

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        api_key=api_key
    )

    workflow = StateGraph(AgentState)

    # Додаємо вузли (агентів)
    workflow.add_node("researcher", partial(researcher_node, llm=llm) )
    workflow.add_node("analyst", partial(analyst_node, llm=llm))
    workflow.add_node("reporter", partial(reporter_node, llm=llm))

    # Визначаємо послідовність виконання
    workflow.set_entry_point("researcher")
    workflow.add_conditional_edges(
        "researcher",
        route_decision_research,
        {
            "analyst": "analyst",
            "end": END,
        }
    )
    # Conditional edges від analyst до researcher або reporter
    workflow.add_conditional_edges(
        "analyst",
        route_after_analyst,
        {
            "researcher": "researcher",
            "reporter": "reporter",
        }
    )


    #workflow.add_edge("analyst", "reporter")
    workflow.add_edge("reporter", END)

    app = workflow.compile()



    # Початковий стан
    initial_state = {
        "topic": "Впровадження ШІ в бізнесс середовищі України",
        "research_results": "",
        "research_attempt": 0,
        "analysis_results": "",
        "analysis_base": 0,
        "final_report": "",
        "final_execution_status": "ERROR",
        "messages": [],
        "timestamp": datetime.now().isoformat()
    }

    try:
        # Запускаємо граф
        final_state = app.invoke(initial_state)

        print("\n" + "="*60)
        print(f"[{final_state["final_execution_status"]}] МУЛЬТИАГЕНТНА СИСТЕМА ЗАВЕРШИЛА РОБОТУ")
        print("="*60)

        # Виводимо повідомлення від агентів
        print("\nЛог виконання:")
        for msg in final_state.get("messages", []):
            print(f"  {msg}")

        # Виводимо фінальний звіт
        print("\n" + final_state["final_report"])

        return final_state

    except Exception as e:
        print(f"\n[ERROR] Помилка виконання: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрограму перервано")
    except Exception as e:
        print(f"\nКритична помилка: {e}")
        import traceback
        traceback.print_exc()