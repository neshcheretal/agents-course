import os
import json
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from ddgs import DDGS
from dotenv import load_dotenv
import unicodedata


# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

@tool("Web_search")
def search_web(query: str) -> str:
    """
    Пошук інформації в інтернеті
    Args:
        query: Текст запиту для пошуку
    """
    print(f"Пошук інформації по темі: {query}")
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(f"- {r['title']}: {r['body'][:150]}...")
        
        return f"Результати пошуку для '{query}':\n" + "\n".join(results)
    except Exception as e:
        print(f" Помилка пошуку: {e}")
        return "Помилка пошуку"


@tool("Current_time")
def get_current_date() -> datetime:
    """
    Отримати поточний час
    """
    today = datetime.today()
    print(f"Поточна дата {today}")
    return today
    

@tool("Data_Analyzer")
def analyze_data(text: str) -> str:
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
"""


@tool("Report_Generator")
def save_report(data: str, filename: str = None) -> str:
    """
    Створення та збереження звіту
    Args:
        data: Дані для збереженя
        filename: Назва файлу для збереження
    """
    if not filename:
        filename = f"crewai_report_{datetime.now():%Y%m%d_%H%M%S}.md"
    
    report_content = f"""
# Звіт CrewAI
**Дата:** {datetime.now():%Y-%m-%d %H:%M}

## Зміст
{data}

---
*Згенеровано CrewAI Multi-Agent System*
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"[OK] Звіт збережено: {filename}")
    return f"[OK] Звіт збережено: {filename}"


def main():
    #Створіть 3 агентів через Agent(role=..., goal=..., backstory=..., tools=...)
    researcher = Agent(
        role="Головний Дослідник AI",
        goal="Знаходити найактуальнішу інформацію з надійних джерел",
        backstory="""
            досвід 15 років у дослідженнях
            Ваша експертиза полягає в глибокому зануренні в теми, пошуку достовірних джерел
        """,
        tools=[search_web, get_current_date],
        verbose=True,
        allow_delegation=False,
        llm="gpt-4.1"
    )


    analytic = Agent(
        role="Старший Аналітик Даних",
        goal="Виявляти закономірності, тренди та інсайти",
        backstory="""
            Ви експерт з data science
            Ваша експертиза полягає в глибокому зануренні в теми та синтезі складної інформації в чіткі, практичні висновки
        """,
        tools=[search_web, analyze_data],
        verbose=True,
        allow_delegation=False,
        llm="gpt-4.1"
    )

    writer = Agent(
        role="Технічний Письменник",
        goal="Створювати зрозумілі та структуровані звіти та зберігати їх у файл",
        backstory="""
            10 років досвіду технічного писання
            Ваш стиль письма чіткий, лаконічний та підкріплений ґрунтовними дослідженнями
        """,
        tools=[save_report, get_current_date],
        verbose=True,
        allow_delegation=False,
        llm="gpt-4.1"
    )


    # TODO: Створіть 3 задачі через Task(description=..., expected_output=..., agent=...)
    research_task = Task(
        description="""
        Проведіть глибоке дослідження на тему {topic}
        
        Використайте Current_time щоб:
        1. Дізнатись поточну дату
        Використайте Web_search Tool для пошуку:
        1. Останніх новин та статей
        2. Статистики та фактів
        3. Думок експертів
        
        Зберіть мінімум 5 ключових фактів. Використовуючи лише публікації не старіші за один рік від поточної дати
        """,
        expected_output="Детальний список знайденої інформації з джерелами",
        agent=researcher
    )

    data_analysis = Task(
        description="""
        Проаналізуйте зібрану інформацію від дослідника.
        
        Використайте Data_Analyzer для:
        1. Виявлення ключових тем
        2. Підрахунку статистики
        3. Визначення трендів
        
        Створіть структурований аналіз з висновками.
        """,
        expected_output="Аналітичний висновок з ключовими інсайтами",
        context=[research_task],
        agent=analytic
    )


    report_task = Task(
        description="""
        Створіть фінальний звіт на основі дослідження та аналізу.
        
        Спочатку використай Current_time щоб:
        1. Дізнатись поточну дату
        Використай Report_Generator для:
        1. Форматування результатів
        2. Створення структурованого документа
        3. Збереження у файл
        
        Звіт має бути зрозумілим для всіх рівнів читачів. 
        Для назви файлу використовуй шаблон "AI-implementation-trends-<Current_time>.txt  де <Current_time> це поточний час отриманий за допогою інструменту Current_time
        """,
        context=[data_analysis],
        expected_output="Професійний звіт збережений у файл з назвою що містить час генерації",
        agent=writer
    )

    # Зберіть команду через Crew(agents=..., tasks=..., process=Process.sequential)
    crew = Crew(
        agents=[researcher, analytic, writer],
        tasks=[research_task, data_analysis, report_task],
        process=Process.sequential,
        verbose=True,
        tracing=False,
        memory=False
    )

    # Запустіть через crew.kickoff()
    result = crew.kickoff(inputs={"topic": unicodedata.normalize("NFC", "The latest AI implementation trends in business environment")})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрограму перервано")
    except Exception as e:
        print(f"\nКритична помилка: {e}")
        import traceback
        traceback.print_exc()