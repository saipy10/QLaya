import 'package:test/test.dart';
import 'package:qlaya/qlaya.dart';

void main() {
  group('QLayaModels', () {
    test('all contains 10 variants', () {
      expect(QLayaModels.all.length, equals(10));
    });

    test('allIds is sorted', () {
      final ids = QLayaModels.allIds;
      expect(ids, equals(List.from(ids)..sort()));
    });

    test('resolve exact key works', () {
      final spec = QLayaModels.resolve('QLaya-TopProduction');
      expect(spec.subfolder, equals('qlaya-int8'));
    });

    test('resolve fuzzy slug works (no prefix, hyphens stripped)', () {
      final spec = QLayaModels.resolve('ultra-fast-edge');
      expect(spec.subfolder, equals('qlaya-distil-6l-int8'));
    });

    test('resolve fuzzy slug case-insensitive', () {
      final spec = QLayaModels.resolve('ULTRASMALLSTORAGE');
      expect(spec.subfolder, equals('qlaya-distil-6l-int4'));
      final spec2 = QLayaModels.resolve('QLaya-UltraSmallStorage');
      expect(spec2.subfolder, equals('qlaya-distil-6l-int4'));
    });

    test('resolve throws ArgumentError for unknown id', () {
      expect(() => QLayaModels.resolve('QLaya-NonExistent'), throwsArgumentError);
    });

    test('all specs have positive sizeMb and latency', () {
      for (final spec in QLayaModels.all.values) {
        expect(spec.sizeMb, greaterThan(0));
        expect(spec.latencyP50Ms, greaterThan(0));
        expect(spec.ramWorkingSetMb, greaterThan(0));
        expect(spec.repo, isNotEmpty);
      }
    });
  });
}
