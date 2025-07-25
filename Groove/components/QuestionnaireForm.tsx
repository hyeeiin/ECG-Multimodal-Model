import React from 'react';
import { View, Text, TextInput, Pressable } from 'react-native';

const questionnaireFields = [
  { key: 'age', label: '나이 (세)' },
  { key: 'weight', label: '체중 (kg)' },
  { key: 'height', label: '키 (cm)' },
];

const checkboxFields = [
  { key: 'hx_stroke', label: '뇌졸중 과거력' },
  { key: 'hx_mi', label: '심근경색 과거력' },
  { key: 'hx_htn', label: '고혈압 과거력' },
  { key: 'hx_dm', label: '당뇨병 과거력' },
  { key: 'hx_dysli', label: '이상지질혈증 과거력' },
  { key: 'hx_athero', label: '중상경화증 과거력' },

  { key: 'fhx_stroke', label: '뇌졸중 가족력' },
  { key: 'fhx_mi', label: '심근경색 가족력' },
  { key: 'fhx_htn', label: '고혈압 가족력' },
  { key: 'fhx_dm', label: '당뇨병 가족력' },
];

const radioFields = [
  {
    key: 'gender',
    label: '성별',
    options: [
      { value: '0', label: '남' },
      { value: '1', label: '여' },
    ],
  },
  {
    key: 'smoke',
    label: '흡연 여부',
    options: [
      { value: '0', label: '무' },
      { value: '1', label: '과거' },
      { value: '2', label: '현재' },
    ],
  },
  {
    key: 'alchol',
    label: '음주 여부',
    options: [
      { value: '0', label: '무' },
      { value: '1', label: '유' },
    ],
  },
  {
    key: 'phy_act',
    label: '운동 여부',
    options: [
      { value: '0', label: '무' },
      { value: '1', label: '저강도' },
      { value: '2', label: '중강도' },
      { value: '3', label: '고강도' },
    ],
  },
];

export default function QuestionnaireForm({ data, onChange }: {
  data: Record<string, string>,
  onChange: (updated: Record<string, string>) => void
}) {

  const handleChange = (key: string, value: string) => {
    onChange({ ...data, [key]: value });
  };

  return (
    <View className='mb-6'>
      {/* 기본 정보 */}
      {questionnaireFields.map((f) => (
        <View key={f.key} className='mb-3'>
          <Text className='font-semibold mb-1'>{f.label}</Text>
          <TextInput
            keyboardType="numeric"
            className='border border-gray-300 rounded px-3 py-2'
            value={data[f.key] || ''}
            onChangeText={(val) => handleChange(f.key, val)}
          />
        </View>
      ))}

      {/* 체크박스 */}
      <Text className='font-bold mt-4 mb-2'>과거력 / 가족력</Text>
      <View className='flex-row flex-wrap'>
        {checkboxFields.map((f) => (
          <Pressable
            key={f.key}
            onPress={() => handleChange(f.key, data[f.key] === '1' ? '0' : '1')}
            className='w-1/2 flex-row items-center mb-2'
          >
            <View className="w-5 h-5 mr-2 border ${data[f.key] === '1' ? 'bg-blue-500' : 'border-gray-400'}" />
            <Text>{f.label}</Text>
          </Pressable>
        ))}
      </View>

      {/* 라디오 그룹 */}
      {radioFields.map((group) => (
        <View key={group.key} className='mt-4'>
          <Text className='font-bold mb-1'>{group.label}</Text>
          <View className='flex-row flex-wrap'>
            {group.options.map((opt) => (
              <Pressable
                key={opt.value}
                onPress={() => handleChange(group.key, opt.value)}
                className='flex-row items-center mr-4 mb-2'
              >
                <View className="w-4 h-4 rounded-full mr-2 border ${data[group.key] === opt.value ? 'bg-blue-500' : 'border-gray-400'}" />
                <Text>{opt.label}</Text>
              </Pressable>
            ))}
          </View>
        </View>
      ))}
    </View>
  );
}
