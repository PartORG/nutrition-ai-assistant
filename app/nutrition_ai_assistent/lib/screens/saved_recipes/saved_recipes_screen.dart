import 'package:flutter/material.dart';
import '../../main.dart';
import '../../services/api_service.dart';
import '../../theme/app_theme.dart';

class SavedRecipesScreen extends StatefulWidget {
  const SavedRecipesScreen({super.key});

  @override
  State<SavedRecipesScreen> createState() => _SavedRecipesScreenState();
}

class _SavedRecipesScreenState extends State<SavedRecipesScreen> {
  bool _loading = true;
  String? _error;
  List<Map<String, dynamic>> _recipes = [];

  @override
  void initState() {
    super.initState();
    _loadRecipes();
  }

  Future<void> _loadRecipes() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final data = await AppServices.instance.api.get('/dashboard')
          as Map<String, dynamic>;
      if (!mounted) return;
      setState(() {
        _recipes = List<Map<String, dynamic>>.from(
            data['recent_recipes'] as List? ?? []);
        _loading = false;
      });
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.message;
        _loading = false;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _error = 'Could not load recipes. Is the server running?';
        _loading = false;
      });
    }
  }

  String _formatDate(String iso) {
    try {
      final dt = DateTime.parse(iso).toLocal();
      final diff = DateTime.now().difference(dt);
      if (diff.inDays == 0) return 'Today';
      if (diff.inDays == 1) return 'Yesterday';
      if (diff.inDays < 7) return '${diff.inDays} days ago';
      return '${dt.day}/${dt.month}/${dt.year}';
    } catch (_) {
      return '';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Saved Recipes'),
        actions: [
          IconButton(
            onPressed: _loadRecipes,
            icon: const Icon(Icons.refresh),
            tooltip: 'Refresh',
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? _ErrorView(message: _error!, onRetry: _loadRecipes)
              : _recipes.isEmpty
                  ? _EmptyView(
                      icon: Icons.bookmark_border,
                      title: 'No saved recipes yet',
                      subtitle:
                          'Ask NutriAI to suggest recipes and save them',
                    )
                  : RefreshIndicator(
                      onRefresh: _loadRecipes,
                      child: ListView.builder(
                        padding: const EdgeInsets.all(12),
                        itemCount: _recipes.length,
                        itemBuilder: (context, i) => _RecipeCard(
                          recipe: _recipes[i],
                          formatDate: _formatDate,
                        ),
                      ),
                    ),
    );
  }
}

// ─── Recipe card ──────────────────────────────────────────────────────────────
class _RecipeCard extends StatefulWidget {
  final Map<String, dynamic> recipe;
  final String Function(String) formatDate;

  const _RecipeCard({required this.recipe, required this.formatDate});

  @override
  State<_RecipeCard> createState() => _RecipeCardState();
}

class _RecipeCardState extends State<_RecipeCard> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final recipe = widget.recipe;
    final name = recipe['recipe_name'] as String? ?? 'Recipe';
    final savedAt = recipe['saved_at'] as String? ?? '';
    final kcal = (recipe['calories'] as num?)?.toDouble();
    final protein = (recipe['protein_g'] as num?)?.toDouble();
    final carbs = (recipe['carbs_g'] as num?)?.toDouble();
    final fat = (recipe['fat_g'] as num?)?.toDouble();
    final fiber = (recipe['fiber_g'] as num?)?.toDouble();
    final sodium = (recipe['sodium_mg'] as num?)?.toDouble();
    final ingredientsRaw = recipe['ingredients'] as String? ?? '';
    final instructions = recipe['cook_instructions'] as String? ?? '';
    final prepTime = recipe['prep_time'] as String? ?? '';
    final servings = recipe['servings'] as int?;

    final ingredients = ingredientsRaw.isNotEmpty
        ? ingredientsRaw.split(',').map((s) => s.trim()).where((s) => s.isNotEmpty).toList()
        : <String>[];
    final hasDetails = ingredients.isNotEmpty || instructions.isNotEmpty;

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: InkWell(
        onTap: hasDetails ? () => setState(() => _expanded = !_expanded) : null,
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // ── Header ────────────────────────────────────────────────────
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: AppColors.cardGreen,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(Icons.restaurant_menu,
                        color: AppColors.primary, size: 22),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          name,
                          style: const TextStyle(
                              fontWeight: FontWeight.bold, fontSize: 15),
                        ),
                        if (savedAt.isNotEmpty)
                          Text(
                            'Saved ${widget.formatDate(savedAt)}',
                            style: const TextStyle(
                                fontSize: 12, color: Colors.grey),
                          ),
                      ],
                    ),
                  ),
                  if (hasDetails)
                    Icon(
                      _expanded ? Icons.expand_less : Icons.expand_more,
                      color: Colors.grey,
                    ),
                ],
              ),

              if (kcal != null) ...[
                const SizedBox(height: 12),
                const Divider(height: 1),
                const SizedBox(height: 12),

                // ── Calorie badge ────────────────────────────────────────────
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                  decoration: BoxDecoration(
                    color: AppColors.primary.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.local_fire_department,
                          color: AppColors.primary, size: 16),
                      const SizedBox(width: 4),
                      Text(
                        '${kcal.toStringAsFixed(0)} kcal',
                        style: const TextStyle(
                          fontWeight: FontWeight.bold,
                          color: AppColors.primaryDark,
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),

                // ── Macro bars ───────────────────────────────────────────────
                if (protein != null)
                  _MacroBar(
                      label: 'Protein',
                      grams: protein,
                      color: AppColors.primary),
                if (carbs != null)
                  _MacroBar(label: 'Carbs', grams: carbs, color: Colors.orange),
                if (fat != null)
                  _MacroBar(
                      label: 'Fat', grams: fat, color: Colors.blue.shade300),
                if (fiber != null && fiber > 0)
                  _MacroBar(
                      label: 'Fiber',
                      grams: fiber,
                      color: Colors.green.shade400),

                // ── Sodium note ──────────────────────────────────────────────
                if (sodium != null && sodium > 0) ...[
                  const SizedBox(height: 6),
                  Row(
                    children: [
                      const Icon(Icons.water_drop_outlined,
                          size: 14, color: Colors.grey),
                      const SizedBox(width: 4),
                      Text(
                        'Sodium: ${sodium.toStringAsFixed(0)} mg',
                        style:
                            const TextStyle(fontSize: 12, color: Colors.grey),
                      ),
                    ],
                  ),
                ],
              ],

              // ── Expandable details ───────────────────────────────────────
              if (_expanded && hasDetails) ...[
                const SizedBox(height: 12),
                const Divider(height: 1),
                const SizedBox(height: 12),

                // Meta row (prep time + servings)
                if (prepTime.isNotEmpty || (servings != null && servings > 0))
                  Padding(
                    padding: const EdgeInsets.only(bottom: 10),
                    child: Wrap(
                      spacing: 12,
                      children: [
                        if (prepTime.isNotEmpty)
                          Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Icon(Icons.timer_outlined,
                                  size: 14, color: Colors.grey),
                              const SizedBox(width: 4),
                              Text(prepTime,
                                  style: const TextStyle(
                                      fontSize: 12, color: Colors.grey)),
                            ],
                          ),
                        if (servings != null && servings > 0)
                          Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Icon(Icons.people_outline,
                                  size: 14, color: Colors.grey),
                              const SizedBox(width: 4),
                              Text('$servings servings',
                                  style: const TextStyle(
                                      fontSize: 12, color: Colors.grey)),
                            ],
                          ),
                      ],
                    ),
                  ),

                // Ingredients
                if (ingredients.isNotEmpty) ...[
                  const Text('Ingredients',
                      style: TextStyle(
                          fontWeight: FontWeight.w600, fontSize: 13)),
                  const SizedBox(height: 6),
                  ...ingredients.map((ing) => Padding(
                        padding: const EdgeInsets.only(bottom: 3),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text('• ',
                                style: TextStyle(
                                    color: AppColors.primary,
                                    fontWeight: FontWeight.bold)),
                            Expanded(
                                child: Text(ing,
                                    style: const TextStyle(fontSize: 13))),
                          ],
                        ),
                      )),
                  const SizedBox(height: 10),
                ],

                // Instructions
                if (instructions.isNotEmpty) ...[
                  const Text('Instructions',
                      style: TextStyle(
                          fontWeight: FontWeight.w600, fontSize: 13)),
                  const SizedBox(height: 6),
                  Text(instructions,
                      style: const TextStyle(fontSize: 13, height: 1.5)),
                ],
              ],
            ],
          ),
        ),
      ),
    );
  }
}

// ─── Macro progress bar ───────────────────────────────────────────────────────
class _MacroBar extends StatelessWidget {
  final String label;
  final double grams;
  final Color color;

  const _MacroBar(
      {required this.label, required this.grams, required this.color});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Container(
              width: 10,
              height: 10,
              decoration:
                  BoxDecoration(color: color, shape: BoxShape.circle)),
          const SizedBox(width: 8),
          SizedBox(
              width: 58,
              child: Text(label, style: const TextStyle(fontSize: 13))),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                // Scale bar to a rough reference maximum of 150 g
                value: (grams / 150).clamp(0.0, 1.0),
                backgroundColor: Colors.grey[200],
                color: color,
                minHeight: 8,
              ),
            ),
          ),
          const SizedBox(width: 8),
          SizedBox(
            width: 52,
            child: Text(
              '${grams.toStringAsFixed(1)} g',
              style: const TextStyle(
                  fontSize: 12, fontWeight: FontWeight.w500),
              textAlign: TextAlign.right,
            ),
          ),
        ],
      ),
    );
  }
}

// ─── Shared helpers ───────────────────────────────────────────────────────────
class _ErrorView extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;
  const _ErrorView({required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.wifi_off, size: 48, color: Colors.grey),
          const SizedBox(height: 16),
          Text(message, style: const TextStyle(color: Colors.grey)),
          const SizedBox(height: 16),
          ElevatedButton.icon(
            onPressed: onRetry,
            icon: const Icon(Icons.refresh),
            label: const Text('Retry'),
          ),
        ],
      ),
    );
  }
}

class _EmptyView extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  const _EmptyView(
      {required this.icon, required this.title, required this.subtitle});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, size: 64, color: Colors.grey[300]),
          const SizedBox(height: 16),
          Text(title,
              style: TextStyle(color: Colors.grey[500], fontSize: 16)),
          const SizedBox(height: 6),
          Text(subtitle,
              style: TextStyle(color: Colors.grey[400], fontSize: 13)),
        ],
      ),
    );
  }
}
