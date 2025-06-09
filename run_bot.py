import os
import time
import logging
import fcntl
import sys
from crypto_bot_ai import CryptoBotAI

def setup_logging():
    """Konfiguracja systemu logowania"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/run_bot.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Główna funkcja uruchamiająca bot"""
    # Sprawdź czy bot nie jest już uruchomiony
    lock_file = open('bot.lock', 'w')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print("Bot jest już uruchomiony!")
        sys.exit(1)

    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Inicjalizacja bota
        bot = CryptoBotAI()
        logger.info("=== BOT ZAINICJALIZOWANY ===")
        
        # Automatyczne uruchomienie bota
        bot.start_bot()
        logger.info("=== BOT URUCHOMIONY AUTOMATYCZNIE ===")
        
        # Połączenie z WebSocket
        if not bot.connect_websocket():
            logger.error("Nie udało się połączyć z WebSocket")
            return
        
        logger.info("Bot uruchomiony pomyślnie")
        
        # Główna pętla
        while bot.is_running:
            try:
                # Automatyczne wykonywanie przewidywań co 5 minut
                current_time = time.time()
                if not hasattr(bot, 'last_prediction_time'):
                    bot.last_prediction_time = 0
                
                if current_time - bot.last_prediction_time >= 300:  # 300 sekund = 5 minut
                    logger.info("=== WYKONUJĘ AUTOMATYCZNE PRZEWIDYWANIA ===")
                    bot.test_predictions()
                    bot.last_prediction_time = current_time
                
                # Sprawdź czy mamy wystarczająco próbek
                if all(len(bot.market_data[pair]) >= 1000 for pair in bot.trading_pairs):
                    logger.info("=== ROZPOCZYNAM TRENOWANIE MODELU ===")
                    bot.train_model()
                    logger.info("=== TRENOWANIE ZAKOŃCZONE ===")
                    break  # Zakończ pętlę po trenowaniu
                
                # Poczekaj 5 minut przed następnym pobraniem
                time.sleep(300)
            except KeyboardInterrupt:
                logger.info("Otrzymano sygnał przerwania")
                break
            except Exception as e:
                logger.error(f"Błąd w głównej pętli: {e}")
                time.sleep(5)
    
    except Exception as e:
        logger.error(f"Błąd krytyczny: {e}")
    finally:
        if 'bot' in locals():
            bot.stop_bot()
        logger.info("Bot zatrzymany")
        # Zwolnij blokadę
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
        try:
            os.remove('bot.lock')
        except:
            pass

if __name__ == "__main__":
    main() 