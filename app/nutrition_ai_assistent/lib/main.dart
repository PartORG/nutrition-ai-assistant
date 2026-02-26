import 'package:flutter/material.dart';
import 'theme/app_theme.dart';
import 'screens/auth/login_screen.dart';
import 'screens/home/home_screen.dart';
import 'services/storage_service.dart';
import 'services/api_service.dart';
import 'services/auth_service.dart';
import 'services/chat_ws_service.dart';

/// Global service instances shared across the app.
class AppServices {
  static AppServices? _instance;

  late final StorageService storage;
  late final ApiService api;
  late final AuthService auth;
  late final ChatWsService chat;

  AppServices._() {
    storage = StorageService();
    api = ApiService(storage);
    auth = AuthService(api, storage);
    chat = ChatWsService(storage);
  }

  static AppServices get instance {
    _instance ??= AppServices._();
    return _instance!;
  }
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final isLoggedIn = await AppServices.instance.auth.isLoggedIn();
  runApp(NutriAIApp(initialRoute: isLoggedIn ? '/home' : '/login'));
}

class NutriAIApp extends StatelessWidget {
  final String initialRoute;

  const NutriAIApp({super.key, required this.initialRoute});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NutriAI',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      initialRoute: initialRoute,
      routes: {
        '/login': (context) => const LoginScreen(),
        '/home': (context) => const HomeScreen(),
      },
    );
  }
}
