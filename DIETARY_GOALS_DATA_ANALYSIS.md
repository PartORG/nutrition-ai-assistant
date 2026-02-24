# Dietary Goals Widget - Data Flow Analysis

## Overview
The dietary goals widget in the dashboard is working correctly to receive data from the database. Below is a detailed analysis of the data flow.

---

## Data Flow

### 1. **Backend (Database → API)**
**Location:** [src/infrastructure/persistence/analytics_repo.py](src/infrastructure/persistence/analytics_repo.py#L96)

The `get_user_dashboard()` method queries:
- **Table:** `nutrition_history`
- **Query Period:** Last 6 days (7 days total including today)
- **Grouping:** `GROUP BY DATE(created_at)` - Daily aggregation

**Returned Data Structure:**
```python
"nutrition_daily": [
    {
        "date": "2026-02-18",
        "calories": 2150.5,
        "protein_g": 85.3,
        "fat_g": 72.1,
        "carbs_g": 245.8,
        "fiber_g": 18.5,
        "sugar_g": 42.0,
        "sodium_mg": 2150.0
    },
    // ... more days
]
```

**Key Fields Used by Dietary Goals Widget:**
- ✅ `sodium_mg` - Used for sodium goal checking
- ✅ `carbs_g` - Used for carbs goal checking
- ✅ `protein_g` - Used for protein goal checking

---

### 2. **API Endpoint**
**Location:** [src/adapters/rest/routers/analytics.py#L35](src/adapters/rest/routers/analytics.py#L35)

- **Endpoint:** `GET /api/dashboard`
- **Returns:** JSON object with `nutrition_daily` array
- **Authentication:** Required (OAuth2)

---

### 3. **Flutter Widget Reception**
**Location:** [app/nutrition_ai_assistent/lib/screens/dashboard/dashboard_screen.dart#L57](app/nutrition_ai_assistent/lib/screens/dashboard/dashboard_screen.dart#L57)

The widget fetches data:
```dart
final data = await api.get('/api/dashboard') as Map<String, dynamic>;
_nutritionDaily = List<Map<String, dynamic>>.from(
  data['nutrition_daily'] as List? ?? [],
);
```

---

## Data Processing in Widget

### Dietary Goals Card Method
**Location:** [dashboard_screen.dart#L245-L310](dashboard_screen.dart#L245-L310)

The `_buildDietaryGoalsCard()` method:

1. **Extracts daily values:**
   ```dart
   wSodium  += (day['sodium_mg']  as num?)?.toDouble() ?? 0;
   wCarbs   += (day['carbs_g']    as num?)?.toDouble() ?? 0;
   wProtein += (day['protein_g']  as num?)?.toDouble() ?? 0;
   ```

2. **Calculates averages:**
   - **Weekly Avg** = sum / total days (7)
   - **Daily Avg** = sum / days with data

3. **Checks against goals:**
   - Sodium: `< 2300 mg/day` (MAX goal)
   - Carbs: `< 300 g/day` (MAX goal)
   - Protein: `> 50 g/day` (MIN goal)

4. **Displays status:**
   - ✅ Green checkmark if goal is met
   - ❌ Red X if goal is not met

---

## ✅ Verification Results

### Data Correctness
| Aspect | Status | Details |
|--------|--------|---------|
| **Database Query** | ✅ Correct | Properly sums nutrition values grouped by date |
| **Field Names** | ✅ Correct | All required fields present (`sodium_mg`, `carbs_g`, `protein_g`) |
| **Data Types** | ✅ Correct | Numbers returned as floats, ready for calculations |
| **Date Range** | ✅ Correct | Last 7 days queried correctly |
| **Null Handling** | ✅ Correct | Defaults to 0 when null using `as num?)?.toDouble() ?? 0` |
| **Calculations** | ✅ Correct | Proper average calculations for weekly and daily stats |

### Potential Issues & Edge Cases

#### 1. **Empty Data Handling** ✅
- If no meals logged: `_nutritionDaily.isEmpty` returns empty list
- Widget displays "—" (dash) for all goals (line 290)
- This is correct behavior

#### 2. **Days with Data vs Total Days**
```dart
daysWithData = count of days where calories > 0
totalDays = _nutritionDaily.length (always 7)
```
- **Daily Avg** uses only days with data
- **Weekly Avg** uses all 7 days
- ✅ This is the intended behavior

#### 3. **Rounding & Precision**
- Backend rounds to 1 decimal place: `round(sr[0] or 0, 1)`
- Frontend converts to `double` for calculations
- Display uses `toStringAsFixed(0)` (no decimals)
- ✅ Acceptable precision for nutrition data

---

## ⚠️ Recommendations

### 1. **Add Logging** (Optional)
For debugging, you could add:
```dart
// In _loadData() after receiving data
print('nutrition_daily length: ${_nutritionDaily.length}');
print('nutrition_daily: $_nutritionDaily');
```

### 2. **Verify Database Has Data**
Check that the database has `nutrition_history` records:
```sql
SELECT COUNT(*), user_id, DATE(created_at) FROM nutrition_history 
GROUP BY user_id, DATE(created_at)
LIMIT 10;
```

### 3. **Test with Sample Data**
Ensure user has logged meals for the last few days to test calculations.

---

## Conclusion

The dietary goals widget is **correctly receiving and processing data** from the database. The data flow is:

```
Database (nutrition_history)
        ↓
Analytics Repo (aggregates daily values)
        ↓
REST API (/api/dashboard)
        ↓
Flutter Widget (_buildDietaryGoalsCard)
        ↓
UI Display (weekly avg, daily avg, goal status)
```

All field mappings, calculations, and null handling are correct. If you're seeing issues, they're likely due to:
1. No nutrition data in the database for the logged-in user
2. Network/API connectivity issues
3. Authentication problems
