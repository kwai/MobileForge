"""
Deterministic Data Module - 确定性数据模块

本模块提供固定的、确定性的数据用于应用数据注入。
使用 AndroidWorld 原生的 sqlite_schema_utils 类型，确保与任务评估系统完全兼容。

关键特性：
- 使用 AndroidWorld 原生 sqlite_schema_utils 类型
- 无随机采样 - 所有数据完整注入
- 固定顺序确保可重复性

参考: reference/MobileForge Emulator Setup/android_world/comprehensive_setup/deterministic_data.py
"""

import datetime
import os
import sys
from typing import Dict, List, Any, Optional

# 将 AndroidWorld 模块路径添加到 sys.path
ANDROID_WORLD_PATH = os.path.join(os.path.dirname(__file__), "models", "AndroidWorld")
if ANDROID_WORLD_PATH not in sys.path:
    sys.path.insert(0, ANDROID_WORLD_PATH)

# 导入 AndroidWorld 原生类型
from android_world.env import device_constants
from android_world.task_evals.utils import sqlite_schema_utils

# 尝试导入 AndroidWorld 原生数据源（用于动态获取数据）
try:
    from android_world.task_evals.single import recipe as recipe_module
    HAS_RECIPE_MODULE = True
except ImportError:
    HAS_RECIPE_MODULE = False

try:
    from android_world.task_evals.information_retrieval import task_app_utils
    HAS_TASK_MODULE = True
except ImportError:
    HAS_TASK_MODULE = False

try:
    from android_world.task_evals.information_retrieval import joplin_app_utils
    HAS_JOPLIN_MODULE = True
except ImportError:
    HAS_JOPLIN_MODULE = False

try:
    from android_world.task_evals.information_retrieval import activity_app_utils
    HAS_ACTIVITY_MODULE = True
except ImportError:
    HAS_ACTIVITY_MODULE = False

# AndroidWorld 标准基准时间
DEVICE_BASE_DATETIME = device_constants.DT


# ============================================================================
# RECIPE DATA (Broccoli App) - 39 固定食谱
# ============================================================================

# 本地食谱数据备份（当无法导入 AndroidWorld 模块时使用）
_LOCAL_RECIPES = [
    {"title": "Spicy Tuna Wraps", "directions": "Mix canned tuna with mayo and sriracha. Spread on tortillas, add lettuce and cucumber slices, roll up."},
    {"title": "Avocado Toast with Egg", "directions": "Toast bread, top with mashed avocado, a fried egg, salt, pepper, and chili flakes."},
    {"title": "Greek Salad Pita Pockets", "directions": "Fill pita pockets with lettuce, cucumber, tomato, feta, olives, and Greek dressing."},
    {"title": "Quick Fried Rice", "directions": "Sauté cooked rice with vegetables, add soy sauce and scrambled eggs. Toss until hot."},
    {"title": "Pesto Pasta with Peas", "directions": "Cook pasta, stir in pesto sauce and cooked peas. Add Parmesan cheese before serving."},
    {"title": "BBQ Chicken Quesadillas", "directions": "Mix shredded cooked chicken with BBQ sauce. Place on tortillas with cheese, fold and cook until crispy."},
    {"title": "Tomato Basil Bruschetta", "directions": "Top sliced baguette with a mix of chopped tomatoes, basil, garlic, olive oil, salt, and pepper."},
    {"title": "Lemon Garlic Tilapia", "directions": "Sauté tilapia in butter, add lemon juice and garlic. Serve with steamed vegetables."},
    {"title": "Turkey and Cheese Panini", "directions": "Layer turkey and cheese on bread, grill in a panini press until golden."},
    {"title": "Veggie and Hummus Sandwich", "directions": "Spread hummus on bread, add cucumber, bell pepper, carrot, and lettuce."},
    {"title": "Mango Chicken Curry", "directions": "Cook chicken pieces in a pan, add onions, garlic, and ginger. Stir in curry powder, coconut milk, and mango pieces. Simmer until chicken is cooked."},
    {"title": "Beef Stir Fry", "directions": "Stir-fry beef slices with broccoli, bell peppers, and onions in soy sauce and garlic. Serve over rice or noodles."},
    {"title": "Shrimp Avocado Salad", "directions": "Mix cooked shrimp with avocado, tomatoes, cucumber, and onion. Dress with lime juice, olive oil, salt, and pepper."},
    {"title": "Spinach and Feta Stuffed Chicken", "directions": "Stuff chicken breasts with a mixture of spinach, feta, garlic, and herbs. Bake until chicken is cooked through."},
    {"title": "Zucchini Noodles with Pesto", "directions": "Spiralize zucchini into noodles, sauté with garlic, then mix with pesto sauce. Top with grated Parmesan cheese."},
    {"title": 'Cauliflower Fried "Rice"', "directions": "Pulse cauliflower in a food processor until it resembles rice. Sauté with vegetables, soy sauce, and add scrambled eggs."},
    {"title": "Sweet Potato and Black Bean Tacos", "directions": "Roast sweet potato cubes, mix with black beans, and use as filling for tacos. Top with avocado and cilantro lime sauce."},
    {"title": "Salmon with Dill Sauce", "directions": "Bake salmon fillets and serve with a sauce made from Greek yogurt, dill, lemon juice, and garlic."},
    {"title": "Quinoa Salad with Vegetables", "directions": "Mix cooked quinoa with diced vegetables, feta cheese, and a lemon olive oil dressing."},
    {"title": "Chickpea Vegetable Soup", "directions": "Sauté onions, carrots, and celery, add broth, canned tomatoes, and chickpeas. Simmer with spinach and seasonings."},
    {"title": "Chicken Caesar Salad Wrap", "directions": "Toss chopped romaine lettuce with Caesar dressing, grilled chicken strips, and Parmesan cheese. Wrap in a large tortilla."},
    {"title": "Vegetarian Chili", "directions": "Cook onions, garlic, bell peppers, and carrots. Add canned tomatoes, kidney beans, black beans, corn, and chili seasoning. Simmer until vegetables are tender."},
    {"title": "Pan-Seared Salmon with Quinoa", "directions": "Pan-sear salmon fillets until crispy. Serve over cooked quinoa with a side of steamed asparagus."},
    {"title": "Caprese Salad Skewers", "directions": "Thread cherry tomatoes, basil leaves, and mozzarella balls onto skewers. Drizzle with balsamic glaze."},
    {"title": "Chicken Alfredo Pasta", "directions": "Cook fettuccine pasta, toss with Alfredo sauce and grilled chicken strips. Serve with a sprinkle of Parmesan cheese."},
    {"title": "Stuffed Bell Peppers", "directions": "Mix cooked quinoa, black beans, corn, tomato sauce, and spices. Stuff into bell peppers and bake until tender."},
    {"title": "Eggplant Parmesan", "directions": "Slice eggplant, bread, and fry. Layer in a baking dish with marinara sauce and mozzarella cheese. Bake until bubbly."},
    {"title": "Thai Peanut Noodle Salad", "directions": "Toss cooked noodles with a Thai peanut sauce, sliced red bell peppers, cabbage, carrots, and cilantro."},
    {"title": "Butternut Squash Soup", "directions": "Sauté onions and garlic, add cubed butternut squash and broth. Puree until smooth and season with nutmeg, salt, and pepper."},
    {"title": "Baked Cod with Lemon and Dill", "directions": "Place cod fillets in a baking dish, season with lemon juice, dill, salt, and pepper. Bake until fish flakes easily."},
    {"title": "Vegetable Stir Fry with Tofu", "directions": "Stir-fry tofu cubes until golden, add assorted vegetables and a stir-fry sauce. Serve over rice or noodles."},
    {"title": "Classic Margherita Pizza", "directions": "Spread pizza dough with tomato sauce, top with slices of mozzarella cheese and fresh basil leaves. Bake until crust is golden."},
    {"title": "Raspberry Almond Smoothie", "directions": "Blend together raspberries, almond milk, banana, and a scoop of almond butter until smooth."},
    {"title": "Moroccan Chickpea Stew", "directions": "Sauté onions, garlic, carrots, and spices. Add canned chickpeas, diced tomatoes, and vegetable broth. Simmer until flavors meld."},
    {"title": "Kale and Quinoa Salad", "directions": "Toss chopped kale, cooked quinoa, dried cranberries, sliced almonds, and feta cheese with a lemon vinaigrette."},
    {"title": "Grilled Cheese with Tomato and Basil", "directions": "Butter bread slices, layer with cheese, tomato slices, and basil. Grill until bread is toasted and cheese is melted."},
    {"title": "Sausage and Peppers Skillet", "directions": "Sauté sliced sausage, bell peppers, and onions until browned. Serve with mustard or on a hoagie roll."},
    {"title": "Lentil Soup", "directions": "Cook onions, carrots, celery, garlic, and lentils in vegetable broth until lentils are tender. Season with thyme and bay leaves."},
    {"title": "Garlic Butter Shrimp", "directions": "Sauté shrimp in butter and minced garlic until pink. Sprinkle with parsley and serve with lemon wedges."},
]

_RECIPE_DESCRIPTIONS = [
    'A quick and easy meal, perfect for busy weekdays.',
    'A delicious and healthy choice for any time of the day.',
    'An ideal recipe for experimenting with different flavors and ingredients.',
]

_SERVINGS_OPTIONS = ['1 serving', '2 servings', '3-4 servings', '6 servings', '8 servings']
_PREP_TIME_OPTIONS = ['10 mins', '20 mins', '30 mins', '45 mins', '1 hrs', '2 hrs']

_INGREDIENT_DESCRIPTORS = [
    'see directions', 'as per recipe', 'varies', 'to preference',
    'quantities to taste', 'as needed', 'optional ingredients',
]

_DIRECTIONS_ADDITIONS = [
    'Try adding a pinch of your favorite spices for extra flavor.',
    'Feel free to substitute with ingredients you have on hand.',
    'Garnish with fresh herbs for a more vibrant taste.',
]


def get_all_recipes() -> List[sqlite_schema_utils.Recipe]:
    """
    获取所有39个固定食谱数据。

    数据源: AndroidWorld recipe._RECIPES 列表
    返回: 使用 sqlite_schema_utils.Recipe 类型的食谱列表
    """
    recipes = []
    
    # 优先使用 AndroidWorld 原生数据
    if HAS_RECIPE_MODULE and hasattr(recipe_module, '_RECIPES'):
        base_recipes = recipe_module._RECIPES
    else:
        # 回退到本地数据
        base_recipes = _LOCAL_RECIPES

    for i, base_recipe in enumerate(base_recipes):
        desc_idx = i % len(_RECIPE_DESCRIPTIONS)
        serv_idx = i % len(_SERVINGS_OPTIONS)
        prep_idx = i % len(_PREP_TIME_OPTIONS)
        ingr_idx = i % len(_INGREDIENT_DESCRIPTORS)
        dir_idx = i % len(_DIRECTIONS_ADDITIONS)

        # 获取 title 和 directions（兼容 dataclass 和 dict）
        if hasattr(base_recipe, 'title'):
            title = base_recipe.title
            directions = base_recipe.directions
        else:
            title = base_recipe["title"]
            directions = base_recipe["directions"]

        # 使用 AndroidWorld 原生 Recipe 类型
        recipe = sqlite_schema_utils.Recipe(
            title=title,
            description=_RECIPE_DESCRIPTIONS[desc_idx],
            servings=_SERVINGS_OPTIONS[serv_idx],
            preparationTime=_PREP_TIME_OPTIONS[prep_idx],
            ingredients=_INGREDIENT_DESCRIPTORS[ingr_idx],
            directions=f'{directions} {_DIRECTIONS_ADDITIONS[dir_idx]}',
        )
        recipes.append(recipe)

    return recipes


# ============================================================================
# TASKS DATA (Tasks App) - 20 固定任务
# ============================================================================

_LOCAL_TASKS = {
    "Grocery Shopping": "Don't forget milk, eggs, and bread. Also need to pick up snacks for the kids.",
    "Finish Project Proposal": "Deadline is Friday. Need to finalize budget and timeline sections.",
    "Schedule Dentist Appointment": "Teeth cleaning overdue. Call Dr. Smith's office.",
    "Water Plants": "Check moisture level before watering. Fertilize succulents.",
    "Meal Prep for the Week": "Make a grocery list based on planned meals. Cook chicken and chop veggies on Sunday.",
    "Research Vacation Destinations": "Looking for beach destinations with family-friendly activities.",
    "Read 'The Martian'": "Started last week. Aim to finish by next weekend.",
    "Call Grandma": "Catch up on family news. Ask for her famous cookie recipe.",
    "Change Air Filter": "Last changed 3 months ago. Buy a new filter at the hardware store.",
    "Brainstorm Blog Post Ideas": "Need 5 new topics for the next month's content calendar.",
    "Renew Driver's License": "Expires next month. Check DMV website for requirements.",
    "Organize Closet": "Donate old clothes and shoes. Put winter clothes in storage.",
    "Submit Expense Report": "Deadline is Wednesday. Attach receipts for all purchases.",
    "Attend Team Meeting": "Agenda includes project updates and brainstorming new initiatives.",
    "Learn to Play Guitar": "Practice chords for 30 minutes every day. Find online tutorials.",
    "Reply to Emails": "Inbox is overflowing. Prioritize urgent messages and unsubscribe from unwanted lists.",
    "Clean Out Fridge": "Check expiration dates and discard old food. Wipe down shelves.",
    "Create Budget for Next Month": "Track income and expenses. Set savings goals.",
    "Back Up Computer Files": "Use external hard drive or cloud storage. Schedule regular backups.",
    "Take Dog to the Vet": "Annual checkup and vaccinations due.",
}


def get_all_tasks() -> List[Dict[str, str]]:
    """
    获取所有20个固定任务数据。

    Returns:
        包含所有20个任务的列表
    """
    # 优先使用 AndroidWorld 原生数据
    if HAS_TASK_MODULE and hasattr(task_app_utils, '_TASKS'):
        tasks_dict = task_app_utils._TASKS
    else:
        tasks_dict = _LOCAL_TASKS
    
    return [{"title": title, "notes": notes} for title, notes in tasks_dict.items()]


def create_deterministic_tasks(base_timestamp_ms: Optional[int] = None) -> List[sqlite_schema_utils.Task]:
    """
    创建确定性的任务数据对象。
    
    与 MobileForge Explore/parallel_exploration/deterministic_data.py 完全一致，
    仅使用 10 个必要字段，确保与 Tasks 应用数据库 schema 兼容。

    Args:
        base_timestamp_ms: 基准时间戳（毫秒）

    Returns:
        sqlite_schema_utils.Task 对象列表
    """
    import uuid

    if base_timestamp_ms is None:
        base_timestamp_ms = int(DEVICE_BASE_DATETIME.timestamp() * 1000)

    tasks = []
    task_data = get_all_tasks()
    
    # 固定的重要性级别分配 - 与 reference 完全一致
    importance_pattern = [0, 1, 2, 3, 2, 1, 0, 3, 2, 1]

    for i, task_info in enumerate(task_data):
        # 确定性时间戳：基于索引计算 - 与 reference 完全一致
        days_offset = (i % 7) + 1  # 1-7天前创建
        hours_offset = (i % 12) + 8  # 8-19点

        created_ts = base_timestamp_ms - (days_offset * 24 * 3600 * 1000) - (hours_offset * 3600 * 1000)
        modified_ts = created_ts + (i * 3600 * 1000)  # 每个任务修改时间递增1小时

        # 前6个任务标记为完成 (30%) - 与 reference 完全一致
        completed_ts = 0
        if i < 6:
            completed_ts = created_ts + (2 * 24 * 3600 * 1000)  # 创建后2天完成

        # 确定性的due date：基于索引 - 与 reference 完全一致
        due_offset_days = (i % 14) - 3  # -3 到 +10 天
        due_ts = base_timestamp_ms + (due_offset_days * 24 * 3600 * 1000)

        # 仅使用 10 个字段，与 reference 完全一致
        task = sqlite_schema_utils.Task(
            title=task_info['title'],
            importance=importance_pattern[i % len(importance_pattern)],
            dueDate=due_ts,
            hideUntil=0,  # 不隐藏
            completed=completed_ts,
            created=created_ts,
            modified=modified_ts,
            notes=task_info['notes'],
            remoteId=str(uuid.UUID(int=i + 1000).int),  # 确定性UUID
            recurrence=None,
        )
        tasks.append(task)

    return tasks


# ============================================================================
# JOPLIN DATA - 12个文件夹，300+ 笔记
# ============================================================================

# Joplin 文件夹数据（本地备份）
_LOCAL_JOPLIN_FOLDERS = {
    "Recipes": [
        {"title": "Zesty Quinoa Salad", "body": "Ingredients:\nCooked quinoa, chopped cucumber, diced tomato, crumbled feta cheese, lemon vinaigrette\nInstructions:\nToss ingredients together. Season to taste."},
        {"title": "Peanut Butter Power Smoothie", "body": "Ingredients:\nPeanut butter, banana, milk of choice, protein powder, ice\nInstructions:\nBlend until smooth and creamy."},
        {"title": "Cheesy Veggie Scramble", "body": "Ingredients:\nEggs, shredded cheese, diced bell pepper, chopped spinach, hot sauce (optional)\nInstructions:\nSauté peppers and spinach. Whisk eggs with cheese, add to pan, and scramble."},
    ],
    "Tasks": [
        {"title": "Morning Routine", "body": "Tasks:\nMake bed\nShower and get dressed\nHealthy breakfast\nReview daily schedule"},
        {"title": "Website Updates", "body": "Tasks:\nAdd new product photos\nUpdate contact form\nFix broken link on About page\nRun website speed test"},
        {"title": "Grocery Trip", "body": "Tasks:\nCheck pantry staples\nMake a list of needed items\nRemember reusable bags\nCheck for coupons or deals"},
    ],
    "Meeting Notes": [
        {"title": "Team Meeting - May 6, 2024", "body": "Agenda, discussion points, action items, decisions made, next steps."},
        {"title": "Client Meeting - Acme Corp. - April 25, 2024", "body": "Attendees, project updates, feedback, next steps, action items."},
        {"title": "Brainstorming Session - New Product Ideas", "body": "Generated ideas, pros and cons, feasibility assessment, next steps."},
    ],
    "Personal": [
        {"title": "Dream Journal Entry", "body": "Had a vivid dream about flying over a vast ocean."},
        {"title": "Bucket List", "body": "1. Learn to surf. 2. Visit Machu Picchu. 3. Write a novel."},
        {"title": "Grocery List", "body": "Milk, eggs, bread, cheese, fruit, vegetables"},
    ],
    "Work": [
        {"title": "Meeting Notes - Q2 Marketing Strategy", "body": "Discussed social media campaigns, new product launch timeline, budget allocation."},
        {"title": "Project Timeline - Website Redesign", "body": "Phase 1: Wireframes due May 15th\nPhase 2: Design approvals by June 1st"},
        {"title": "Performance Review Talking Points", "body": "- Exceeded sales targets by 15%\n- Successfully led cross-functional team"},
    ],
    "School": [
        {"title": "Lecture Notes - Intro to Psychology", "body": "Key concepts: nature vs. nurture, cognitive development, social psychology."},
        {"title": "Reading List - American Literature", "body": "- The Scarlet Letter\n- The Great Gatsby\n- Moby Dick"},
        {"title": "Study Guide - Calculus Midterm", "body": "Topics covered: derivatives, integrals, limits, applications."},
    ],
    "Home": [
        {"title": "Home Maintenance Schedule", "body": "Spring: clean gutters, check roof for damage, service AC"},
        {"title": "Recipe - Chicken Noodle Soup", "body": "Ingredients: chicken, noodles, carrots, celery, onion, broth, herbs."},
        {"title": "Cleaning Checklist", "body": "Kitchen: clean countertops, wipe down appliances, sweep and mop floor"},
    ],
    "Projects": [
        {"title": "Community Garden Project", "body": "Create a shared green space for the neighborhood, promoting sustainable food production."},
        {"title": "Home Renovation - Kitchen Remodel", "body": "Design plans, budget, materials list, contractor quotes, timeline."},
        {"title": "Mobile App Development - Expense Tracker", "body": "Project outline, wireframes, technology stack, development timeline."},
    ],
    "Ideas": [
        {"title": "Personalized Pet Portraits", "body": "Offer custom-painted portraits of pets based on photos provided by clients."},
        {"title": "Language Learning App", "body": "Gamified language learning app with interactive exercises and personalized feedback."},
        {"title": "Sustainable Fashion Subscription Box", "body": "Curated selection of eco-friendly clothing and accessories delivered monthly."},
    ],
    "Health": [
        {"title": "Workout Routine - Strength Training", "body": "Exercises for each muscle group, sets, reps, rest periods, weekly schedule."},
        {"title": "Meal Plan - Week of May 6th", "body": "Breakfast, lunch, dinner, snacks for each day, grocery list, recipes."},
        {"title": "Doctor's Appointment Notes", "body": "Summary of discussion with doctor, diagnosis, treatment plan, medication list."},
    ],
    "Travel": [
        {"title": "Trip Itinerary - Europe Summer 2024", "body": "Flights, accommodations, transportation, daily activities, sightseeing plans."},
        {"title": "Packing List - Beach Vacation", "body": "Clothing, toiletries, electronics, travel documents, beach gear, first-aid kit."},
        {"title": "Travel Budget - Southeast Asia", "body": "Estimated costs for flights, accommodation, food, transportation, activities."},
    ],
    "Finance": [
        {"title": "Monthly Budget - May 2024", "body": "Income, expenses, savings goals, spending categories, debt repayment plan."},
        {"title": "Investment Portfolio Summary", "body": "Breakdown of investments (stocks, bonds, mutual funds), performance overview."},
        {"title": "Retirement Savings Plan", "body": "Contribution schedule, target retirement age, projected retirement income."},
    ],
}


def get_all_joplin_folders() -> List[str]:
    """获取所有 Joplin 文件夹名称。"""
    if HAS_JOPLIN_MODULE and hasattr(joplin_app_utils, '_FOLDERS'):
        return list(joplin_app_utils._FOLDERS.keys())
    return list(_LOCAL_JOPLIN_FOLDERS.keys())


def get_all_joplin_notes() -> Dict[str, List[Dict[str, str]]]:
    """获取所有 Joplin 笔记数据。"""
    if HAS_JOPLIN_MODULE and hasattr(joplin_app_utils, '_FOLDERS'):
        return joplin_app_utils._FOLDERS.copy()
    return _LOCAL_JOPLIN_FOLDERS.copy()


def get_total_joplin_notes_count() -> int:
    """获取 Joplin 笔记总数。"""
    folders = get_all_joplin_notes()
    return sum(len(notes) for notes in folders.values())


# ============================================================================
# OPENTRACKS DATA - 16个运动类别
# ============================================================================

_ACTIVITY_CATEGORIES = [
    "biking", "running", "hiking", "swimming", "walking", "skiing",
    "snowboarding", "kayaking", "rowing", "sailing", "skateboarding",
    "surfing", "climbing", "mountain biking", "road biking", "trail running"
]

_ACTIVITY_NAMES = {
    "biking": ["Morning Bike Ride", "Evening Cycling", "City Tour"],
    "running": ["Morning Run", "Evening Jog", "Interval Training"],
    "hiking": ["Mountain Trail", "Forest Walk", "Scenic Hike"],
    "swimming": ["Pool Laps", "Open Water Swim", "Training Session"],
    "walking": ["Park Walk", "City Stroll", "Nature Walk"],
    "skiing": ["Downhill Run", "Cross Country", "Powder Day"],
    "snowboarding": ["Half Pipe", "Backcountry", "Park Session"],
    "kayaking": ["River Paddle", "Lake Tour", "Sea Kayaking"],
    "rowing": ["Morning Row", "Training Session", "Race Practice"],
    "sailing": ["Bay Cruise", "Racing", "Leisure Sail"],
    "skateboarding": ["Street Session", "Park Tricks", "Cruising"],
    "surfing": ["Dawn Patrol", "Afternoon Session", "Sunset Surf"],
    "climbing": ["Bouldering", "Lead Climbing", "Top Rope"],
    "mountain biking": ["Trail Ride", "Downhill", "Cross Country"],
    "road biking": ["Training Ride", "Group Ride", "Time Trial"],
    "trail running": ["Forest Trail", "Mountain Run", "Technical Trail"],
}


def get_all_activity_categories() -> List[str]:
    """获取所有运动类别。"""
    if HAS_ACTIVITY_MODULE and hasattr(activity_app_utils, '_CATEGORY_TO_ACTIVITY_NAMES'):
        return list(activity_app_utils._CATEGORY_TO_ACTIVITY_NAMES.keys())
    return _ACTIVITY_CATEGORIES.copy()


def get_activity_names_by_category() -> Dict[str, List[str]]:
    """获取每个运动类别下的活动名称。"""
    if HAS_ACTIVITY_MODULE and hasattr(activity_app_utils, '_CATEGORY_TO_ACTIVITY_NAMES'):
        return activity_app_utils._CATEGORY_TO_ACTIVITY_NAMES.copy()
    return _ACTIVITY_NAMES.copy()


def get_all_activity_data() -> List[Dict[str, Any]]:
    """
    获取所有活动的原始数据，包含详细的字段信息。
    """
    activities = []
    categories = get_all_activity_categories()
    activity_names = get_activity_names_by_category()
    
    activity_index = 0
    for cat_idx, category in enumerate(categories):
        names = activity_names.get(category, [f"{category} Activity"])
        num_activities = 2 + (cat_idx % 2)
        
        for name_idx in range(min(num_activities, len(names))):
            name = names[name_idx]
            
            duration_minutes = 30 + (activity_index * 10) % 150
            duration_ms = duration_minutes * 60 * 1000
            distance = 1000 + (activity_index * 500) % 14000
            elevation_gain = (activity_index * 20) % 500
            min_elevation = 50 + (activity_index * 10) % 200
            max_elevation = min_elevation + elevation_gain
            
            activity = {
                'name': name,
                'category': category,
                'description': f'Deterministic {category} activity #{activity_index + 1}',
                'totaldistance': float(distance),
                'totaltime': duration_ms,
                'movingtime': duration_ms,
                'elevationgain': float(elevation_gain),
                'minelevation': float(min_elevation),
                'maxelevation': float(max_elevation),
            }
            activities.append(activity)
            activity_index += 1
    
    return activities


def create_deterministic_activities(base_timestamp_ms: Optional[int] = None) -> List[sqlite_schema_utils.SportsActivity]:
    """
    创建确定性的运动活动数据。

    Args:
        base_timestamp_ms: 基准时间戳（毫秒）

    Returns:
        sqlite_schema_utils.SportsActivity 对象列表
    """
    if base_timestamp_ms is None:
        base_timestamp_ms = int(DEVICE_BASE_DATETIME.timestamp() * 1000)

    activities = []
    categories = get_all_activity_categories()
    activity_names = get_activity_names_by_category()

    activity_index = 0
    for cat_idx, category in enumerate(categories):
        names = activity_names.get(category, [f"{category} Activity"])
        num_activities = 2 + (cat_idx % 2)

        for name_idx in range(min(num_activities, len(names))):
            name = names[name_idx]

            days_ago = activity_index + 1
            start_hour = 6 + (activity_index % 12)

            start_ts = base_timestamp_ms - (days_ago * 24 * 3600 * 1000)
            start_ts += start_hour * 3600 * 1000

            duration_minutes = 30 + (activity_index * 10) % 150
            duration_ms = duration_minutes * 60 * 1000
            stop_ts = start_ts + duration_ms

            distance = 1000 + (activity_index * 500) % 14000
            elevation_gain = (activity_index * 20) % 500
            elevation_loss = (activity_index * 15) % 400
            avg_speed = distance / (duration_ms / 1000) if duration_ms > 0 else 0
            max_speed = avg_speed * 1.5

            # 使用 AndroidWorld 原生 SportsActivity 类型
            # 不手动设置 uuid，让 dataclass 使用默认的 uuid4().bytes
            activity = sqlite_schema_utils.SportsActivity(
                name=name,
                category=category,
                activity_type=category,
                description=f'Deterministic {category} activity #{activity_index + 1}',
                totaldistance=float(distance),
                starttime=start_ts,
                stoptime=stop_ts,
                numpoints=int(duration_ms / 10000),
                totaltime=duration_ms,
                movingtime=duration_ms,
                avgspeed=avg_speed,
                avgmovingspeed=avg_speed,
                maxspeed=max_speed,
                minelevation=50.0 + (activity_index * 10) % 200,
                maxelevation=100.0 + (activity_index * 10) % 200 + elevation_gain,
                elevationgain=float(elevation_gain),
                elevationloss=float(elevation_loss),
            )
            activities.append(activity)
            activity_index += 1

    return activities


# ============================================================================
# CALENDAR DATA - 25个固定日历事件
# ============================================================================

_CALENDAR_EVENT_TEMPLATES = [
    ('Team Meeting', 'meeting', 60, 'Conference Room A'),
    ('Project Review', 'meeting', 90, 'Meeting Room B'),
    ('Client Call', 'meeting', 30, 'Phone'),
    ('Doctor Appointment', 'appointment', 45, 'Medical Center'),
    ('Dentist Checkup', 'appointment', 60, 'Dental Clinic'),
    ('Birthday Party', 'birthday', 1440, 'Home'),
    ('Anniversary', 'birthday', 1440, ''),
    ('Pay Bills', 'reminder', 0, ''),
    ('Submit Report', 'reminder', 0, ''),
    ('Gym Session', 'task', 90, 'Fitness Center'),
    ('Weekly Review', 'meeting', 60, 'Office'),
    ('Sprint Planning', 'meeting', 120, 'Conference Room'),
    ('Code Review', 'meeting', 45, 'Virtual'),
    ('Lunch Meeting', 'meeting', 60, 'Restaurant'),
    ('Training Session', 'meeting', 180, 'Training Room'),
    ('Interview', 'appointment', 60, 'HR Office'),
    ('Car Service', 'appointment', 120, 'Auto Shop'),
    ('Mom Birthday', 'birthday', 1440, ''),
    ('Project Deadline', 'reminder', 0, ''),
    ('Team Building', 'meeting', 240, 'Event Hall'),
    ('Quarterly Review', 'meeting', 90, 'Board Room'),
    ('Standup Meeting', 'meeting', 15, 'Virtual'),
    ('Design Review', 'meeting', 60, 'Design Lab'),
    ('Performance Review', 'appointment', 45, 'Manager Office'),
    ('Yoga Class', 'task', 60, 'Yoga Studio'),
]


def create_deterministic_calendar_events(base_timestamp: Optional[int] = None) -> List[sqlite_schema_utils.CalendarEvent]:
    """
    创建确定性的日历事件数据。

    Args:
        base_timestamp: 基准Unix时间戳（秒）

    Returns:
        sqlite_schema_utils.CalendarEvent 对象列表
    """
    if base_timestamp is None:
        base_timestamp = int(DEVICE_BASE_DATETIME.timestamp())

    events = []

    for i, (title, event_type, duration_minutes, location) in enumerate(_CALENDAR_EVENT_TEMPLATES):
        day_offset = (i % 21) - 7
        hour = 8 + (i % 10)

        start_ts = base_timestamp + (day_offset * 24 * 3600) + (hour * 3600)
        end_ts = start_ts + duration_minutes * 60

        # 使用 AndroidWorld 原生 CalendarEvent 类型
        event = sqlite_schema_utils.CalendarEvent(
            start_ts=start_ts,
            end_ts=end_ts,
            title=title,
            location=location,
            description=f'Deterministic {event_type} event #{i + 1}',
            repeat_interval=0,
            repeat_rule=0,
            reminder_1_minutes=-1,
            reminder_2_minutes=-1,
            reminder_3_minutes=-1,
            reminder_1_type=0,
            reminder_2_type=0,
            reminder_3_type=0,
            repeat_limit=0,
            repetition_exceptions='[]',
            attendees='',
            import_id='',
            time_zone=device_constants.TIMEZONE,  # 使用 AndroidWorld 标准时区
            flags=0,
            event_type=1,
            parent_id=0,
            last_updated=0,
            source='comprehensive-setup',
            availability=0,
            color=0,
            type=0
        )
        events.append(event)

    return events


# ============================================================================
# EXPENSE DATA (Pro Expense) - 30条固定费用记录
# 与 reference/MobileForge Explore/parallel_exploration/deterministic_data.py 完全一致
# Category ID 映射:
#   1: Food, 2: Transport, 3: Entertainment, 4: Health, 5: Bills,
#   6: Shopping, 7: Education, 8: Other, 9: Savings
# ============================================================================

_EXPENSE_TEMPLATES = [
    ('Groceries', 1, 45.99),       # 1 = Food
    ('Gas', 2, 52.30),             # 2 = Transport
    ('Restaurant', 1, 28.50),      # 1 = Food
    ('Coffee', 1, 5.25),           # 1 = Food
    ('Movie Tickets', 3, 24.00),   # 3 = Entertainment
    ('Gym Membership', 4, 49.99),  # 4 = Health
    ('Phone Bill', 5, 85.00),      # 5 = Bills
    ('Internet', 5, 65.00),        # 5 = Bills
    ('Electricity', 5, 120.50),    # 5 = Bills
    ('Water Bill', 5, 35.00),      # 5 = Bills
    ('Uber', 2, 18.75),            # 2 = Transport
    ('Amazon', 6, 156.99),         # 6 = Shopping
    ('Netflix', 3, 15.99),         # 3 = Entertainment
    ('Spotify', 3, 9.99),          # 3 = Entertainment
    ('Lunch', 1, 12.50),           # 1 = Food
    ('Dinner', 1, 35.00),          # 1 = Food
    ('Books', 7, 29.99),           # 7 = Education
    ('Clothing', 6, 89.99),        # 6 = Shopping
    ('Medicine', 4, 22.50),        # 4 = Health
    ('Haircut', 8, 25.00),         # 8 = Other
    ('Parking', 2, 8.00),          # 2 = Transport
    ('Subway', 2, 2.75),           # 2 = Transport
    ('Snacks', 1, 6.50),           # 1 = Food
    ('Office Supplies', 7, 34.99), # 7 = Education
    ('Gift', 8, 50.00),            # 8 = Other
    ('Dry Cleaning', 8, 18.00),    # 8 = Other
    ('Pet Food', 8, 42.00),        # 8 = Other
    ('Insurance', 5, 150.00),      # 5 = Bills
    ('Rent', 5, 1500.00),          # 5 = Bills
    ('Savings', 9, 500.00),        # 9 = Savings
]

# Category 名称映射（与 reference/MobileForge Explore 完全一致）
_EXPENSE_CATEGORY_NAMES = {
    1: 'Food',
    2: 'Transport',
    3: 'Entertainment',
    4: 'Health',
    5: 'Bills',
    6: 'Shopping',
    7: 'Education',
    8: 'Other',
    9: 'Savings',
}


def create_deterministic_expenses(base_timestamp_ms: Optional[int] = None) -> List[sqlite_schema_utils.Expense]:
    """
    创建确定性的费用记录数据。
    
    与 reference/MobileForge Explore/parallel_exploration/deterministic_data.py 完全一致。
    返回 sqlite_schema_utils.Expense 对象列表。
    
    Category ID 映射:
        1: Food, 2: Transport, 3: Entertainment, 4: Health, 5: Bills,
        6: Shopping, 7: Education, 8: Other, 9: Savings

    Args:
        base_timestamp_ms: 基准时间戳（毫秒）

    Returns:
        sqlite_schema_utils.Expense 对象列表
    """
    if base_timestamp_ms is None:
        base_timestamp_ms = int(DEVICE_BASE_DATETIME.timestamp() * 1000)

    expenses = []

    for i, (name, category_id, amount) in enumerate(_EXPENSE_TEMPLATES):
        days_ago = i % 30
        created_date = base_timestamp_ms - (days_ago * 24 * 3600 * 1000)
        modified_date = created_date

        # 返回 sqlite_schema_utils.Expense 对象
        expense = sqlite_schema_utils.Expense(
            name=name,
            amount=int(amount * 100),  # 以分为单位
            category=category_id,
            note=f'Deterministic expense #{i + 1}',
            created_date=created_date,
            modified_date=modified_date,
        )
        expenses.append(expense)

    return expenses


# ============================================================================
# CONTACTS DATA - 50个固定联系人
# ============================================================================

DETERMINISTIC_CONTACTS = [
    {'name': 'Alice Johnson', 'phone': '+1-555-0101'},
    {'name': 'Bob Smith', 'phone': '+1-555-0102'},
    {'name': 'Carol Williams', 'phone': '+1-555-0103'},
    {'name': 'David Brown', 'phone': '+1-555-0104'},
    {'name': 'Emily Davis', 'phone': '+1-555-0105'},
    {'name': 'Frank Miller', 'phone': '+1-555-0106'},
    {'name': 'Grace Wilson', 'phone': '+1-555-0107'},
    {'name': 'Henry Moore', 'phone': '+1-555-0108'},
    {'name': 'Ivy Taylor', 'phone': '+1-555-0109'},
    {'name': 'Jack Anderson', 'phone': '+1-555-0110'},
    {'name': 'Kate Thomas', 'phone': '+1-555-0111'},
    {'name': 'Leo Jackson', 'phone': '+1-555-0112'},
    {'name': 'Mia White', 'phone': '+1-555-0113'},
    {'name': 'Noah Harris', 'phone': '+1-555-0114'},
    {'name': 'Olivia Martin', 'phone': '+1-555-0115'},
    {'name': 'Peter Garcia', 'phone': '+1-555-0116'},
    {'name': 'Quinn Martinez', 'phone': '+1-555-0117'},
    {'name': 'Rachel Robinson', 'phone': '+1-555-0118'},
    {'name': 'Sam Clark', 'phone': '+1-555-0119'},
    {'name': 'Tina Rodriguez', 'phone': '+1-555-0120'},
    {'name': 'Uma Lewis', 'phone': '+1-555-0121'},
    {'name': 'Victor Lee', 'phone': '+1-555-0122'},
    {'name': 'Wendy Walker', 'phone': '+1-555-0123'},
    {'name': 'Xavier Hall', 'phone': '+1-555-0124'},
    {'name': 'Yolanda Allen', 'phone': '+1-555-0125'},
    {'name': 'Zack Young', 'phone': '+1-555-0126'},
    {'name': 'Amy King', 'phone': '+1-555-0127'},
    {'name': 'Brian Wright', 'phone': '+1-555-0128'},
    {'name': 'Cindy Scott', 'phone': '+1-555-0129'},
    {'name': 'Daniel Green', 'phone': '+1-555-0130'},
    {'name': 'Eva Adams', 'phone': '+1-555-0131'},
    {'name': 'Fred Baker', 'phone': '+1-555-0132'},
    {'name': 'Gina Nelson', 'phone': '+1-555-0133'},
    {'name': 'Howard Hill', 'phone': '+1-555-0134'},
    {'name': 'Iris Ramirez', 'phone': '+1-555-0135'},
    {'name': 'James Campbell', 'phone': '+1-555-0136'},
    {'name': 'Kelly Mitchell', 'phone': '+1-555-0137'},
    {'name': 'Larry Roberts', 'phone': '+1-555-0138'},
    {'name': 'Mary Carter', 'phone': '+1-555-0139'},
    {'name': 'Nick Phillips', 'phone': '+1-555-0140'},
    {'name': 'Oscar Evans', 'phone': '+1-555-0141'},
    {'name': 'Paula Turner', 'phone': '+1-555-0142'},
    {'name': 'Quentin Torres', 'phone': '+1-555-0143'},
    {'name': 'Rita Parker', 'phone': '+1-555-0144'},
    {'name': 'Steve Collins', 'phone': '+1-555-0145'},
    {'name': 'Tracy Edwards', 'phone': '+1-555-0146'},
    {'name': 'Ulysses Stewart', 'phone': '+1-555-0147'},
    {'name': 'Vera Sanchez', 'phone': '+1-555-0148'},
    {'name': 'Will Morris', 'phone': '+1-555-0149'},
    {'name': 'Xena Rogers', 'phone': '+1-555-0150'},
]


def get_all_contacts() -> List[Dict[str, str]]:
    """获取所有50个固定联系人数据。"""
    return DETERMINISTIC_CONTACTS.copy()


# ============================================================================
# MARKOR DATA - 10个固定 Markdown 文件
# ============================================================================

DETERMINISTIC_MARKOR_DOCUMENTS = [
    {
        'filename': 'meeting_notes.md',
        'content': '''# Meeting Notes

## Project Status Meeting - October 2023

### Attendees
- Alice Johnson
- Bob Smith
- Carol Williams

### Agenda
1. Project timeline review
2. Resource allocation
3. Next steps

### Action Items
- [ ] Complete design review by Friday
- [ ] Schedule client call
- [ ] Update documentation
'''
    },
    {
        'filename': 'todo_list.md',
        'content': '''# To-Do List

## Work Tasks
- [x] Review pull request
- [ ] Update API documentation
- [ ] Fix bug in login module
- [ ] Prepare presentation slides

## Personal Tasks
- [ ] Buy groceries
- [ ] Schedule dentist appointment
- [ ] Call mom
'''
    },
    {
        'filename': 'project_ideas.md',
        'content': '''# Project Ideas

## Mobile App Concepts

### 1. Fitness Tracker
Track daily workouts, calories, and progress

### 2. Recipe Manager
Save and organize favorite recipes

### 3. Budget Tracker
Monitor expenses and savings goals

## Web Projects
- Portfolio website
- Blog platform
- E-commerce template
'''
    },
    {
        'filename': 'shopping_list.md',
        'content': '''# Shopping List

## Groceries
- Milk
- Eggs
- Bread
- Cheese
- Fruits
- Vegetables

## Household Items
- Soap
- Toothpaste
- Paper towels
- Cleaning supplies

## Electronics
- USB cable
- Phone charger
'''
    },
    {
        'filename': 'study_notes.md',
        'content': '''# Study Notes

## Python Programming

### Data Types
- int, float, str, bool
- list, tuple, dict, set

### Control Flow
- if/elif/else
- for loops
- while loops

### Functions
```python
def greet(name):
    return f"Hello, {name}!"
```

## Important Concepts
1. Object-Oriented Programming
2. Error Handling
3. File I/O
'''
    },
    {
        'filename': 'health_log.md',
        'content': '''# Health Log

## Weekly Exercise
| Day | Activity | Duration |
|-----|----------|----------|
| Mon | Running  | 30 min   |
| Wed | Gym      | 45 min   |
| Fri | Swimming | 60 min   |
| Sat | Hiking   | 2 hours  |

## Nutrition Goals
- Drink 8 glasses of water daily
- Eat 5 servings of fruits/vegetables
- Limit processed foods
'''
    },
    {
        'filename': 'travel_plans.md',
        'content': '''# Travel Plans

## Japan Trip - Spring 2024

### Itinerary
- Day 1-3: Tokyo
- Day 4-5: Kyoto
- Day 6: Nara
- Day 7: Osaka

### Budget
- Flights: $1,200
- Accommodation: $800
- Food: $500
- Activities: $300

### Packing List
- Passport
- Adapters
- Comfortable shoes
'''
    },
    {
        'filename': 'book_reviews.md',
        'content': '''# Book Reviews

## Currently Reading
**The Pragmatic Programmer** by David Thomas

Rating: 5/5 stars

Key Takeaways:
1. DRY - Don't Repeat Yourself
2. Orthogonality in design
3. Tracer bullets

## To Read
- Clean Code
- Design Patterns
- Refactoring
'''
    },
    {
        'filename': 'recipes.md',
        'content': '''# Favorite Recipes

## Pasta Carbonara

### Ingredients
- 400g spaghetti
- 200g pancetta
- 4 eggs
- 100g parmesan
- Black pepper

### Instructions
1. Cook pasta al dente
2. Fry pancetta until crispy
3. Mix eggs and cheese
4. Combine all ingredients
5. Season with pepper
'''
    },
    {
        'filename': 'work_notes.md',
        'content': '''# Work Notes

## Q4 Goals
1. Complete mobile app v2.0
2. Improve test coverage to 80%
3. Reduce bug backlog by 50%

## Team Updates
- New developer joining next week
- Sprint review on Friday
- Holiday schedule reminder

## Technical Debt
- Refactor authentication module
- Update deprecated dependencies
- Improve error logging
'''
    },
]


def get_all_markor_documents() -> List[Dict[str, str]]:
    """获取所有10个固定 Markor 文档数据。"""
    return DETERMINISTIC_MARKOR_DOCUMENTS.copy()


# ============================================================================
# SMS 数据 - 使用与 MobileForge Explore 完全相同的生成逻辑
# ============================================================================

def get_all_sms_messages(base_timestamp_ms: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    获取确定性的短信对话数据。
    
    使用与 MobileForge Explore 完全相同的生成逻辑：
    - 使用 sms_templates 和前 10 个联系人循环生成
    - 每个联系人创建 3-5 条消息
    
    Args:
        base_timestamp_ms: 基准时间戳（毫秒）
        
    Returns:
        短信对话列表
    """
    if base_timestamp_ms is None:
        base_timestamp_ms = int(DEVICE_BASE_DATETIME.timestamp() * 1000)
    
    conversations = []
    
    # 使用前10个联系人创建对话 - 与 MobileForge Explore 完全一致
    sms_templates = [
        ("Hey, how are you?", True),
        ("I'm good, thanks! How about you?", False),
        ("Great! Want to grab lunch tomorrow?", True),
        ("Sure, sounds good. Where?", False),
        ("How about the Italian place on Main St?", True),
        ("Perfect! See you at noon.", False),
        ("Don't forget about the meeting today", True),
        ("Thanks for the reminder!", False),
        ("Can you send me the report?", True),
        ("Sending it now.", False),
        ("Got it, thanks!", True),
        ("Happy birthday!", True),
        ("Thank you so much!", False),
        ("Are you free this weekend?", True),
        ("Yes, what did you have in mind?", False),
    ]
    
    for i, contact in enumerate(DETERMINISTIC_CONTACTS[:10]):
        # 每个联系人创建3-5条消息
        num_messages = 3 + (i % 3)
        for j in range(num_messages):
            msg_idx = (i * 3 + j) % len(sms_templates)
            message, is_incoming = sms_templates[msg_idx]
            
            # 确定性时间戳
            days_ago = (i + j) % 14
            hours_ago = (i * 2 + j) % 24
            timestamp = base_timestamp_ms - (days_ago * 24 * 3600 * 1000) - (hours_ago * 3600 * 1000)
            
            conversations.append({
                'address': contact['phone'],
                'contact_name': contact['name'],
                'body': message,
                'type': 1 if is_incoming else 2,  # 1=incoming, 2=outgoing
                'date': timestamp,
                'read': 1,
            })
    
    return conversations


# ============================================================================
# 通话记录数据 - 使用与 MobileForge Explore 完全相同的生成逻辑
# ============================================================================

def get_all_call_history(base_timestamp_ms: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    获取确定性的通话记录数据。
    
    使用与 MobileForge Explore 完全相同的生成逻辑：
    - 使用 call_types 和 durations 固定数组
    - 使用前 15 个联系人
    - 每个联系人 1-2 条通话记录
    
    Args:
        base_timestamp_ms: 基准时间戳（毫秒）
        
    Returns:
        通话记录列表
    """
    if base_timestamp_ms is None:
        base_timestamp_ms = int(DEVICE_BASE_DATETIME.timestamp() * 1000)
    
    call_history = []
    
    # 通话类型: 1=incoming, 2=outgoing, 3=missed - 与 MobileForge Explore 完全一致
    call_types = [1, 2, 1, 3, 2, 1, 2, 3, 1, 2]
    durations = [120, 300, 60, 0, 180, 240, 90, 0, 150, 420]  # 未接电话时长为0
    
    for i, contact in enumerate(DETERMINISTIC_CONTACTS[:15]):
        # 每个联系人1-2条通话记录
        num_calls = 1 + (i % 2)
        for j in range(num_calls):
            call_idx = (i + j) % len(call_types)
            call_type = call_types[call_idx]
            duration = durations[call_idx] if call_type != 3 else 0
            
            # 确定性时间戳
            days_ago = (i * 2 + j) % 21
            hours_ago = (i + j * 3) % 24
            timestamp = base_timestamp_ms - (days_ago * 24 * 3600 * 1000) - (hours_ago * 3600 * 1000)
            
            call_history.append({
                'number': contact['phone'],
                'contact_name': contact['name'],
                'type': call_type,
                'duration': duration,
                'date': timestamp,
            })
    
    return call_history


# ============================================================================
# VLC 视频文件 - 使用与 MobileForge Explore 相同的生成逻辑
# ============================================================================

# 固定的视频文件数量
DETERMINISTIC_VLC_VIDEOS_COUNT = 15


def get_all_vlc_videos() -> List[str]:
    """
    获取所有固定的 VLC 视频文件名。
    
    使用与 MobileForge Explore 相同的生成逻辑：
    video_types = ['clip', 'footage', 'scene', 'recording', 'highlight']
    filename = f'{video_type}_{i+1:02d}.mp4'
    """
    video_types = ['clip', 'footage', 'scene', 'recording', 'highlight']
    videos = []
    for i in range(15):
        video_type = video_types[i % len(video_types)]
        videos.append(f'{video_type}_{i+1:02d}.mp4')
    return videos


def get_deterministic_video_files() -> List[Dict[str, Any]]:
    """
    获取确定性的视频文件数据。
    
    与 MobileForge Explore/parallel_exploration/deterministic_data.py 完全一致。
    
    Returns:
        视频文件数据列表，每个包含 filename, title, duration_seconds, directory, messages
    """
    video_types = ['clip', 'footage', 'scene', 'recording', 'highlight']
    
    videos = []
    for i in range(15):
        video_type = video_types[i % len(video_types)]
        videos.append({
            'filename': f'{video_type}_{i+1:02d}.mp4',
            'title': f'{video_type.title()} {i+1}',
            'duration_seconds': 20 + (i * 10) % 160,  # 20-180秒
            'directory': '/storage/emulated/0/VLCVideos',
            'messages': [f'Video Content {i+1}', 'Sample Video'],  # 与 reference 一致，2个元素
        })
    
    return videos


# ============================================================================
# MUSIC DATA (Retro Music) - 固定音乐数据
# 与 reference/MobileForge Explore/parallel_exploration/deterministic_data.py 对齐
# ============================================================================

def get_deterministic_music_files() -> list:
    """
    获取确定性的音乐文件数据。
    
    与 reference/MobileForge Explore/parallel_exploration/deterministic_data.py
    中的 get_deterministic_music_files() 完全一致，确保注入数据的确定性。
    
    Returns:
        音乐文件数据列表
    """
    artists = ['Artist Alpha', 'Artist Beta', 'Artist Gamma', 'Artist Delta']
    albums = ['Album One', 'Album Two', 'Album Three']
    
    music_files = []
    song_idx = 0
    
    for artist_idx, artist in enumerate(artists):
        album = albums[artist_idx % len(albums)]
        # 每个艺术家3-4首歌
        num_songs = 3 + (artist_idx % 2)
        for i in range(num_songs):
            music_files.append({
                'filename': f'{artist.replace(" ", "_")}_song_{i+1:02d}.mp3',
                'title': f'Song {song_idx + 1}',
                'artist': artist,
                'album': album,
                'duration_ms': 180000 + (song_idx * 30000) % 120000,  # 3-5分钟
                'directory': '/storage/emulated/0/Music',
            })
            song_idx += 1
    
    return music_files


# ============================================================================
# 数据摘要统计
# ============================================================================

def print_data_summary():
    """打印所有固定数据的统计摘要。"""
    print("=" * 50)
    print("MobileForge Deterministic Data Summary")
    print("=" * 50)
    print(f"Recipes (Broccoli):     {len(get_all_recipes())} items")
    print(f"Tasks:                  {len(get_all_tasks())} items")
    print(f"Joplin Folders:         {len(get_all_joplin_folders())} folders")
    print(f"Joplin Notes:           {get_total_joplin_notes_count()} notes")
    print(f"Activity Categories:    {len(get_all_activity_categories())} categories")
    print(f"Calendar Events:        {len(_CALENDAR_EVENT_TEMPLATES)} events")
    print(f"Expenses:               {len(_EXPENSE_TEMPLATES)} records")
    print(f"Contacts:               {len(DETERMINISTIC_CONTACTS)} contacts")
    print(f"Markor Documents:       {len(DETERMINISTIC_MARKOR_DOCUMENTS)} documents")
    print(f"VLC Videos:             {DETERMINISTIC_VLC_VIDEOS_COUNT} videos")
    print(f"Music Files:            {len(get_deterministic_music_files())} files")
    print("=" * 50)


if __name__ == '__main__':
    print_data_summary()
