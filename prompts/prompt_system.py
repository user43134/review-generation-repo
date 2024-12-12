class PromptSystem:
    """Работа с систем промптом"""
    
    @staticmethod
    def get_system_prompt():
        return (
            "Ты эксперт в области написания отзывов о товарах, заведениях и услугах. "
            "Твоя задача — писать информативные, честные и убедительные отзывы. "
            "Ты должен учитывать предоставленный контекст, чтобы отзыв выглядел естественно и правдоподобно."
        )

    @staticmethod
    def generate_user_prompt(category, rating, keywords, similar_reviews):
        reviews_text = "\n".join([f"{i+1}. {text}" for i, text in enumerate(similar_reviews)])
        return f"""
        Вот несколько отзывов о похожих местах:
        {reviews_text}

        Создай новый отзыв о месте с характеристиками: 
        - Категория: {category}
        - Рейтинг: {rating}
        - Ключевые слова: {', '.join(keywords)}

        Новый отзыв:
        """