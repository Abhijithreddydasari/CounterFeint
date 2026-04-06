"""
Fraud and legitimate ad templates for synthetic data generation.

Templates are organized by difficulty: obvious scams for Task 1,
sophisticated scams for Task 2, and network-ready scams for Task 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class AdTemplate:
    category: str
    ad_copies: List[str]
    targeting_hints: List[str]
    risk_signals: List[str]
    label: str  # "fraud", "legit", or "escalate"
    fraud_type: str = ""
    severity: float = 0.0  # 0.0-1.0
    difficulty: str = "easy"  # easy, medium, hard


LEGIT_TEMPLATES: List[AdTemplate] = [
    AdTemplate(
        category="ecommerce",
        ad_copies=[
            "Spring Sale — Up to 30% off home essentials. Free shipping on orders over $50. Shop now at HomeNest.",
            "New arrivals in women's fashion. Curated styles for every occasion. Visit StyleHaven today.",
            "Upgrade your kitchen with premium cookware. Professional-grade, lifetime warranty. Chef's Choice Store.",
            "Organic skincare products made with natural ingredients. Dermatologist tested. GlowNatural.com",
            "Handcrafted leather bags — designed to last. Each piece unique. ArtisanHide.com",
        ],
        targeting_hints=[
            "Women 25-54, interests: home decor, shopping",
            "Women 18-45, interests: fashion, lifestyle",
            "Adults 30-60, interests: cooking, kitchen",
            "Women 22-50, interests: beauty, wellness",
            "Adults 25-55, interests: fashion, luxury goods",
        ],
        risk_signals=[],
        label="legit",
        severity=0.0,
    ),
    AdTemplate(
        category="saas",
        ad_copies=[
            "Streamline your project management. 14-day free trial, no credit card required. Try TaskFlow Pro.",
            "Automate your invoicing in minutes. Trusted by 50,000+ small businesses. InvoiceSimple.",
            "HR management made easy. Onboarding, payroll, compliance — all in one platform. PeopleFirst.",
            "Secure cloud backup for your business. 256-bit encryption. Plans from $9/mo. VaultBackup.",
            "Email marketing that converts. AI-powered subject lines, drag-and-drop editor. MailCraft.",
        ],
        targeting_hints=[
            "Professionals 25-55, interests: business, productivity",
            "Small business owners, interests: finance, accounting",
            "HR managers 30-50, interests: management, recruiting",
            "IT professionals 25-55, interests: cloud, security",
            "Marketers 25-45, interests: digital marketing, email",
        ],
        risk_signals=[],
        label="legit",
        severity=0.0,
    ),
    AdTemplate(
        category="local_service",
        ad_copies=[
            "Need a plumber? Same-day service, licensed & insured. Call Mike's Plumbing — serving Portland since 2008.",
            "Award-winning pizza delivery in Brooklyn. Family recipes since 1985. Order from Sal's Pizzeria.",
            "Professional house cleaning. Eco-friendly products, background-checked staff. SparkleHome Services.",
            "Expert tax preparation. CPA with 15 years experience. Free consultation. Williams Tax Group.",
            "Dog walking and pet sitting. GPS-tracked walks, daily photo updates. PawPals Chicago.",
        ],
        targeting_hints=[
            "Homeowners 30-65, Portland metro area",
            "Adults 18-55, Brooklyn, interests: food delivery",
            "Homeowners 25-60, interests: home services",
            "Adults 25-65, tax season, interests: finance",
            "Pet owners 25-50, Chicago metro area",
        ],
        risk_signals=[],
        label="legit",
        severity=0.0,
    ),
    AdTemplate(
        category="education",
        ad_copies=[
            "Learn Python programming in 12 weeks. Live instructor, project-based curriculum. CodeAcademy Pro.",
            "MBA programs online — AACSB accredited. Flexible schedule for working professionals. StateU Online.",
            "SAT prep that works. Average 150-point score improvement. Guaranteed. PrepMaster Tutoring.",
            "Professional photography workshops. Small class sizes, hands-on practice. LensArt Studio.",
            "Spanish fluency in 6 months. Native instructors, conversation-first approach. LinguaViva.",
        ],
        targeting_hints=[
            "Adults 18-35, interests: programming, tech careers",
            "Professionals 25-45, interests: MBA, business education",
            "Parents of teens, interests: college prep, SAT",
            "Adults 25-55, interests: photography, creative arts",
            "Adults 18-50, interests: language learning, travel",
        ],
        risk_signals=[],
        label="legit",
        severity=0.0,
    ),
    AdTemplate(
        category="fitness",
        ad_copies=[
            "Join our community gym. No contracts, cancel anytime. First month $19.99. FitLife Fitness.",
            "Yoga for beginners — online classes with certified instructors. 7-day free trial. ZenFlow Yoga.",
            "Custom meal plans designed by registered dietitians. Achieve your health goals. NutriPlan.",
            "Running shoes engineered for comfort. 90-day return policy. SwiftStride Athletics.",
            "Personal training at home. Certified trainers, flexible scheduling. FitVisit.",
        ],
        targeting_hints=[
            "Adults 18-55, interests: fitness, gym",
            "Adults 25-50, interests: yoga, wellness, mindfulness",
            "Health-conscious adults 22-55, interests: nutrition, diet",
            "Runners 18-55, interests: running, athletics",
            "Adults 25-50, interests: personal training, fitness",
        ],
        risk_signals=[],
        label="legit",
        severity=0.0,
    ),
]

FRAUD_TEMPLATES: List[AdTemplate] = [
    # === OBVIOUS SCAMS (Task 1) ===
    AdTemplate(
        category="fake_giveaway",
        ad_copies=[
            "CONGRATULATIONS!! You've been selected to win a FREE iPhone 16 Pro!! Click NOW before offer expires!!!",
            "🎉 WINNER ALERT 🎉 Claim your FREE $1000 Amazon Gift Card! Limited time only! Act NOW!",
            "You are the 1,000,000th visitor! Click here to claim your FREE MacBook Pro! Don't miss out!!!",
            "FREE PS5 GIVEAWAY! Just complete a quick survey! Only 3 left! Hurry!!!",
            "EXCLUSIVE: Win a Tesla Model 3! Enter your email NOW! Drawing ends tonight!!!",
        ],
        targeting_hints=[
            "All adults 18+, broad targeting, worldwide",
            "Adults 18-65, interests: shopping, deals, coupons",
            "Adults 18-35, interests: technology, gaming",
            "Young adults 16-30, interests: gaming, PlayStation",
            "Adults 18-55, interests: cars, Tesla, electric vehicles",
        ],
        risk_signals=[
            "excessive urgency language",
            "too-good-to-be-true offer",
            "excessive punctuation/caps",
            "no purchase necessary claims",
        ],
        label="fraud",
        fraud_type="fake_giveaway",
        severity=0.9,
        difficulty="easy",
    ),
    AdTemplate(
        category="counterfeit_goods",
        ad_copies=[
            "Designer handbags 90% OFF! Gucci, Louis Vuitton, Prada — all authentic! Limited stock! BargainLux.shop",
            "Rolex watches starting at $49.99! 100% genuine! Free worldwide shipping! LuxWatchDeals.net",
            "Nike Air Jordan CLEARANCE — $29.99! All sizes available! Original quality guaranteed! SneakerVault.store",
            "Ray-Ban sunglasses factory direct! 95% discount! Premium quality! SunDealz.com",
            "Canada Goose jackets $79.99! Authentic merchandise, warehouse clearance! ArcticDeals.shop",
        ],
        targeting_hints=[
            "Adults 18-45, interests: luxury brands, fashion deals",
            "Men 25-55, interests: watches, luxury",
            "Young adults 16-30, interests: sneakers, streetwear",
            "Adults 18-50, interests: fashion, sunglasses, deals",
            "Adults 20-45, interests: outdoor gear, fashion",
        ],
        risk_signals=[
            "impossibly low prices for luxury goods",
            "unfamiliar domain",
            "claims of authenticity with extreme discounts",
        ],
        label="fraud",
        fraud_type="counterfeit",
        severity=0.8,
        difficulty="easy",
    ),
    AdTemplate(
        category="miracle_cure",
        ad_copies=[
            "DOCTORS HATE THIS! One simple trick cures diabetes in 7 days! Big Pharma doesn't want you to know!",
            "Lose 30 pounds in 2 weeks with this ancient herb! No diet, no exercise! MiracleSlim pills!",
            "CURE CANCER NATURALLY! This secret remedy has a 99% success rate! Order now before it's banned!",
            "Reverse aging in 14 days! Scientists discover fountain of youth pill! Limited supply!",
            "Grow 3 inches taller at ANY age! Patented growth formula! 100% guaranteed results!",
        ],
        targeting_hints=[
            "Adults 35-70, interests: health, diabetes, natural remedies",
            "Adults 25-60, interests: weight loss, diet, fitness",
            "Adults 40-75, interests: cancer, health, alternative medicine",
            "Adults 35-65, interests: anti-aging, beauty, health",
            "Men 18-40, interests: fitness, height, self-improvement",
        ],
        risk_signals=[
            "unsubstantiated medical claims",
            "conspiracy language ('they don't want you to know')",
            "impossible health promises",
            "urgency/scarcity tactics",
        ],
        label="fraud",
        fraud_type="miracle_cure",
        severity=0.95,
        difficulty="easy",
    ),
    AdTemplate(
        category="advance_fee",
        ad_copies=[
            "URGENT: I am Prince Adebayo of Nigeria. I need your help to transfer $4.5 million. You keep 30%.",
            "Congratulations! You've inherited $2.8 million from a distant relative. Pay $200 processing fee to claim.",
            "WORK FROM HOME! Earn $5000/week stuffing envelopes! No experience needed! Send $50 registration fee.",
            "URGENT BUSINESS PROPOSAL: Oil company executive needs foreign partner. $10M deal. Reply immediately.",
            "You've won the UK National Lottery! Send $100 for tax processing to receive your $1.5M prize.",
        ],
        targeting_hints=[
            "All adults 18+, broad targeting",
            "Adults 30-70, interests: finance, inheritance",
            "Adults 18-65, interests: work from home, jobs",
            "Adults 30-65, interests: business, investment",
            "Adults 25-70, interests: lottery, gambling, luck",
        ],
        risk_signals=[
            "advance fee request",
            "unsolicited money offer",
            "urgency pressure",
            "request for personal information",
        ],
        label="fraud",
        fraud_type="advance_fee_scam",
        severity=0.85,
        difficulty="easy",
    ),

    # === SOPHISTICATED SCAMS (Task 2) ===
    AdTemplate(
        category="fake_crypto",
        ad_copies=[
            "Invest in DeFi with confidence. Our AI-powered trading platform delivers 12-15% monthly returns. Audited smart contracts. Join CryptoYield Pro.",
            "Stake your crypto with institutional-grade security. 8% APY on BTC, 12% on ETH. Regulated in Singapore. VaultStake Finance.",
            "New token launch: GreenChain ($GRNC) — carbon credit blockchain. Pre-sale live. Backed by major VCs. Read our whitepaper.",
            "Automated crypto arbitrage — our algorithm exploits price differences across exchanges. Average 20% monthly. QuantBot Trading.",
            "Web3 real estate tokenization. Own fractions of premium properties. Starting at $100. Licensed broker-dealer. PropChain.io",
        ],
        targeting_hints=[
            "Men 25-45, interests: cryptocurrency, DeFi, investing",
            "Adults 25-55, interests: crypto staking, passive income",
            "Tech-savvy adults 20-40, interests: blockchain, Web3, tokens",
            "Adults 25-50, interests: algo trading, crypto, finance",
            "Adults 30-55, interests: real estate, investing, crypto",
        ],
        risk_signals=[
            "high guaranteed returns",
            "complex financial product",
        ],
        label="fraud",
        fraud_type="fake_crypto",
        severity=0.7,
        difficulty="medium",
    ),
    AdTemplate(
        category="celebrity_endorsement_fraud",
        ad_copies=[
            "Elon Musk reveals his #1 investment for 2026. 'This will be bigger than Tesla.' Exclusive interview inside.",
            "Warren Buffett's secret stock pick leaked! Financial advisors are scrambling. See the details before it's removed.",
            "Oprah's favorite weight loss secret finally revealed. 'I lost 40 pounds without dieting.' Doctor-approved method.",
            "Mark Cuban: 'I'm putting everything into this AI company.' Find out which stock billionaires are buying.",
            "Taylor Swift partners with new wellness brand. 'It changed my life.' Limited celebrity edition available.",
        ],
        targeting_hints=[
            "Adults 25-55, interests: investing, Elon Musk, technology",
            "Adults 35-70, interests: stocks, Warren Buffett, investing",
            "Women 25-60, interests: weight loss, Oprah, wellness",
            "Adults 25-50, interests: AI, investing, Mark Cuban",
            "Adults 18-45, interests: Taylor Swift, wellness, beauty",
        ],
        risk_signals=[
            "celebrity name used without verification",
            "urgency/exclusivity language",
        ],
        label="fraud",
        fraud_type="fake_endorsement",
        severity=0.65,
        difficulty="medium",
    ),
    AdTemplate(
        category="clone_brand",
        ad_copies=[
            "Amaz0n Warehouse Clearance — Electronics up to 70% off. Official overstock sale. Limited time. amaz0n-deals.com",
            "PayPaI Security Alert: Your account needs verification. Update your information to avoid suspension. Secure portal.",
            "Walmrt+ Members: Exclusive flash sale this weekend only. Save an extra 25% on groceries. Members-only link.",
            "App1e Trade-In Event: Get up to $800 for your old device. Upgrade to iPhone 16. Authorized reseller.",
            "Netfliix Premium — Get 1 year for $29.99! Special promotional price. Activate through our partner link.",
        ],
        targeting_hints=[
            "Adults 18-65, interests: Amazon, deals, electronics",
            "Adults 18-65, interests: online payments, PayPal",
            "Adults 25-60, interests: Walmart, grocery shopping",
            "Adults 18-55, interests: Apple, iPhone, technology",
            "Adults 18-50, interests: Netflix, streaming, entertainment",
        ],
        risk_signals=[
            "brand name misspelling",
            "unofficial domain",
        ],
        label="fraud",
        fraud_type="brand_impersonation",
        severity=0.75,
        difficulty="medium",
    ),
    AdTemplate(
        category="gray_area_supplements",
        ad_copies=[
            "NeuroPeak — clinically studied nootropic for enhanced focus and memory. 30-day money-back guarantee. FDA-registered facility.",
            "TestoMax Elite — natural testosterone support for men over 40. Doctor-formulated. Published clinical trial results.",
            "SlimBiome Pro — proprietary probiotic blend for healthy weight management. Featured in Health Magazine. GMP certified.",
            "Joint Restore Plus — advanced collagen peptide formula. Thousands of 5-star reviews. Orthopedist recommended.",
            "SleepDeep Ultra — melatonin-free natural sleep aid. Non-habit forming. Money-back guarantee. PharmGrade Labs.",
        ],
        targeting_hints=[
            "Adults 25-55, interests: nootropics, productivity, brain health",
            "Men 40-65, interests: men's health, testosterone, fitness",
            "Adults 30-60, interests: weight loss, probiotics, gut health",
            "Adults 45-75, interests: joint health, supplements, arthritis",
            "Adults 30-65, interests: sleep, insomnia, wellness",
        ],
        risk_signals=[
            "supplement with strong efficacy claims",
            "medical-sounding language",
        ],
        label="escalate",
        fraud_type="gray_area",
        severity=0.4,
        difficulty="medium",
    ),

    # === NETWORK SCAMS (Task 3 — individual ads that belong to fraud rings) ===
    # These templates look almost entirely legitimate on the surface.
    # The fraud is only detectable through cross-ad investigation.
    AdTemplate(
        category="network_crypto",
        ad_copies=[
            "Earn passive income with our DeFi staking pool. Community-driven, transparent governance. BlockYield.io",
            "Next-gen DEX with lowest fees. Trade with zero slippage on major pairs. SwapNova Protocol.",
            "NFT marketplace for digital artists. Mint, list, and sell with 2% fees. Creator-first platform. ArtBlock.exchange",
            "Cross-chain bridge with military-grade security. Move assets between chains in seconds. ChainLink Pro.",
            "Crypto tax software — auto-import from 50+ exchanges. CPA-approved reports. TaxChain Solutions.",
        ],
        targeting_hints=[
            "Adults 20-40, interests: DeFi, crypto staking",
            "Adults 20-45, interests: crypto trading, DEX",
            "Adults 20-40, interests: NFT, digital art, crypto",
            "Adults 25-45, interests: cross-chain, crypto bridges",
            "Adults 25-55, interests: crypto, tax, finance",
        ],
        risk_signals=[],
        label="fraud",
        fraud_type="coordinated_network",
        severity=0.6,
        difficulty="hard",
    ),
    AdTemplate(
        category="network_ecommerce",
        ad_copies=[
            "Premium wireless earbuds — ANC, 40hr battery. Compare to AirPods Pro at half the price. SoundElite Store.",
            "Smart home security camera — 2K resolution, night vision, cloud storage. TechGuard Home.",
            "Portable power station for camping. 1000W, solar-ready. Adventure-proof design. PowerTrail Gear.",
            "Ergonomic office chair — lumbar support, breathable mesh. 5-year warranty. WorkComfort Direct.",
            "Robot vacuum with LiDAR mapping. Self-emptying dock. Smart home compatible. CleanBot Pro.",
        ],
        targeting_hints=[
            "Adults 18-45, interests: tech, audio, headphones",
            "Adults 25-55, interests: home security, smart home",
            "Adults 25-50, interests: camping, outdoor gear",
            "Adults 25-55, interests: home office, ergonomics",
            "Adults 25-55, interests: smart home, cleaning",
        ],
        risk_signals=[],
        label="fraud",
        fraud_type="coordinated_network",
        severity=0.5,
        difficulty="hard",
    ),
    AdTemplate(
        category="network_fintech",
        ad_copies=[
            "Simplify your bookkeeping. AI-powered categorization, bank sync, tax-ready reports. LedgerFlow.",
            "Expense management for growing teams. Real-time tracking, receipt scanning, approval workflows. SpendWise.",
            "Invoicing made simple. Send professional invoices in 30 seconds. Accept 40+ payment methods. BillStream.",
            "Payroll for startups — automated tax filings, direct deposit, compliance included. PayRun HR.",
            "Business credit monitoring. Real-time alerts, score tracking, credit-building tips. BizCredit Insights.",
        ],
        targeting_hints=[
            "Small business owners 28-55, interests: accounting, bookkeeping",
            "Finance managers 30-50, interests: expense management, SaaS",
            "Freelancers 22-45, interests: invoicing, self-employment",
            "Startup founders 25-45, interests: payroll, HR tech",
            "Entrepreneurs 30-55, interests: business credit, finance",
        ],
        risk_signals=[],
        label="fraud",
        fraud_type="coordinated_network",
        severity=0.55,
        difficulty="hard",
    ),
    AdTemplate(
        category="network_health",
        ad_copies=[
            "Telehealth appointments in minutes. Board-certified physicians, prescriptions delivered. MediConnect Health.",
            "Mental wellness platform. Licensed therapists, flexible scheduling, insurance accepted. MindBridge Care.",
            "At-home lab testing. Comprehensive panels, fast results, doctor-reviewed. HealthCheck Direct.",
            "Nutrition coaching backed by science. Personalized meal plans from registered dietitians. NutriGuide Pro.",
            "Fitness tracking reimagined. AI-powered workout plans, progress analytics, community support. FitMetrics.",
        ],
        targeting_hints=[
            "Adults 25-60, interests: telemedicine, healthcare",
            "Adults 25-50, interests: mental health, therapy, wellness",
            "Health-conscious adults 30-55, interests: lab testing, preventive health",
            "Adults 25-50, interests: nutrition, diet planning, health",
            "Fitness enthusiasts 20-45, interests: fitness tracking, wearables",
        ],
        risk_signals=[],
        label="fraud",
        fraud_type="coordinated_network",
        severity=0.5,
        difficulty="hard",
    ),
]
