import os
import stripe
from dotenv import load_dotenv

load_dotenv()  # Only needed locally or if you use .env in dev

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY")
