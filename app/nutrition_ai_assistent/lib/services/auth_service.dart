import 'api_service.dart';
import 'storage_service.dart';

/// Wraps register / login / logout API calls.
class AuthService {
  final ApiService _api;
  final StorageService _storage;

  AuthService(this._api, this._storage);

  Future<void> login(String username, String password) async {
    final data = await _api.post('/auth/login', {
      'login': username,
      'password': password,
    });
    await _storage.saveAuthData(
      token: data['access_token'] as String,
      userId: data['user_id'] as int,
      username: username,
    );
  }

  Future<void> register({
    required String name,
    required String surname,
    required String username,
    required String password,
    int age = 0,
    String gender = '',
    String caretaker = '',
    String healthCondition = '',
  }) async {
    final data = await _api.post('/auth/register', {
      'name': name,
      'surname': surname,
      'login': username,
      'password': password,
      'age': age,
      'gender': gender,
      'caretaker': caretaker,
      'health_condition': healthCondition,
    });
    await _storage.saveAuthData(
      token: data['access_token'] as String,
      userId: data['user_id'] as int,
      username: username,
      name: '$name $surname',
    );
  }

  Future<void> logout() async {
    await _storage.clearAuthData();
  }

  Future<bool> isLoggedIn() async => _storage.hasToken();
}
