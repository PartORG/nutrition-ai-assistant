import 'dart:convert';
import 'package:http/http.dart' as http;
import 'storage_service.dart';

class ApiException implements Exception {
  final int statusCode;
  final String message;
  ApiException(this.statusCode, this.message);

  @override
  String toString() => message;
}

/// Base HTTP client. The API URL is set at compile time via --dart-define.
///
/// Usage:
///   flutter run                                               → http://localhost:8000 (default)
///   flutter run --dart-define=API_BASE_URL=http://10.0.2.2:8000   → Android emulator
///   flutter run --dart-define=API_BASE_URL=http://192.168.1.x:8000 → Physical device (LAN IP)
///   flutter run --dart-define=API_BASE_URL=https://api.myapp.com   → Production
class ApiService {
  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://127.0.0.1:8000',
  );

  final StorageService _storage;

  ApiService(this._storage);

  Future<Map<String, String>> _authHeaders() async {
    final token = await _storage.getToken();
    return {
      'Content-Type': 'application/json',
      if (token != null) 'Authorization': 'Bearer $token',
    };
  }

  Future<dynamic> post(
    String path,
    Map<String, dynamic> body, {
    bool auth = false,
  }) async {
    final headers = auth
        ? await _authHeaders()
        : {'Content-Type': 'application/json'};
    final response = await http
        .post(
          Uri.parse('$baseUrl$path'),
          headers: headers,
          body: jsonEncode(body),
        )
        .timeout(const Duration(seconds: 30));
    return _handle(response);
  }

  /// Upload an image and return the server-side path.
  ///
  /// Accepts raw bytes + filename so this works on all platforms including Web
  /// (dart:io is not available in browsers, so fromPath is forbidden).
  /// Callers should read bytes via XFile.readAsBytes() before calling this.
  Future<String> uploadImageBytes(List<int> bytes, String filename) async {
    final token = await _storage.getToken();
    final request = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/upload/image'),
    );
    if (token != null) {
      request.headers['Authorization'] = 'Bearer $token';
    }
    request.files.add(http.MultipartFile.fromBytes('file', bytes, filename: filename));
    final streamed = await request.send().timeout(const Duration(seconds: 30));
    final response = await http.Response.fromStream(streamed);
    final data = _handle(response);
    return data['path'] as String;
  }

  Future<dynamic> get(String path) async {
    final headers = await _authHeaders();
    final response = await http
        .get(Uri.parse('$baseUrl$path'), headers: headers)
        .timeout(const Duration(seconds: 30));
    return _handle(response);
  }

  dynamic _handle(http.Response response) {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      if (response.body.isEmpty) return null;
      return jsonDecode(utf8.decode(response.bodyBytes));
    }
    dynamic body = {};
    try {
      body = jsonDecode(response.body);
    } catch (_) {}
    final detail = body['detail'] ?? 'Request failed (${response.statusCode})';
    throw ApiException(response.statusCode, detail.toString());
  }
}
