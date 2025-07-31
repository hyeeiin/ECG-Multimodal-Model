import React from 'react'
import {
  YStack,
  XStack,
  Input,
  Label,
  Text,
  Checkbox,
  RadioGroup,
  Card,
} from 'tamagui'
import { Check } from '@tamagui/lucide-icons'
import { TouchableOpacity } from 'react-native'

type Props = {
  data: Record<string, string>
  onChange: (updated: Record<string, string>) => void
}

const questionnaireFields = [
  { key: 'age', label: '나이 (세)' },
  { key: 'height', label: '키 (cm)' },
  { key: 'weight', label: '체중 (kg)' },
]

const sexGroups = [
  {
    key: 'gender',
    label: '성별',
    options: [
      { value: '0', label: '남' },
      { value: '1', label: '여' },
    ],
  }
]

const pastHistoryFields = [
  { key: 'hx_stroke', label: '뇌졸중 과거력' },
  { key: 'hx_mi', label: '심근경색 과거력' },
  { key: 'hx_htn', label: '고혈압 과거력' },
  { key: 'hx_dm', label: '당뇨병 과거력' },
  { key: 'hx_dysli', label: '이상지질혈증 과거력' },
  { key: 'hx_athero', label: '중상경화증 과거력' },
]

const familyHistoryFields = [
  { key: 'fhx_stroke', label: '뇌졸중 가족력' },
  { key: 'fhx_mi', label: '심근경색 가족력' },
  { key: 'fhx_htn', label: '고혈압 가족력' },
  { key: 'fhx_dm', label: '당뇨병 가족력' },
]

const radioGroups = [
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
    key: 'alcohol',
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
]

export default function QuestionnaireForm({ data, onChange }: Props) {
  const handleChange = (key: string, value: string) => {
    onChange({ ...data, [key]: value })
  }

  return (
    <YStack space="$4">
      {/* 기본 정보 */}
      <Card bordered padded elevate>
        <YStack space="$3">
          <Text fontSize="$5" fontWeight="700">
            기본 정보
          </Text>
          {questionnaireFields.map((f) => (
            <YStack key={f.key}>
              <Label>{f.label}</Label>
              <Input
                keyboardType="numeric"
                value={data[f.key] || ''}
                onChangeText={(val) => handleChange(f.key, val)}
              />
            </YStack>
          ))}
        </YStack>
      </Card>

      {/* 성별 */}
      {sexGroups.map((group) => (
        <Card key={group.key} bordered padded elevate>
          <YStack space="$4">
            <Text fontSize="$5" fontWeight="700">
              {group.label}
            </Text>
            <RadioGroup
              value={data[group.key] || ''}
              onValueChange={(val) => handleChange(group.key, val)}
            >
              <YStack space="$2">
                {group.options.map((opt) => (
                  <TouchableOpacity
                    key={opt.value}
                    onPress={() => handleChange(group.key, opt.value)}
                  >
                    <XStack space="$2" style={{ alignItems: 'center' }}>
                      <RadioGroup.Item value={opt.value} size="$3">
                        <RadioGroup.Indicator />
                      </RadioGroup.Item>
                      <Label>{opt.label}</Label>
                    </XStack>
                  </TouchableOpacity>
                ))}
              </YStack>
            </RadioGroup>
          </YStack>
        </Card>
      ))}

      {/* 과거력 */}
      <Card bordered padded elevate>
        <YStack space="$4">
          <Text fontSize="$5" fontWeight="700">
            과거력
          </Text>
          {pastHistoryFields.map((f) => {
            const isChecked = data[f.key] === '1'
            return (
              <TouchableOpacity
                key={f.key}
                onPress={() => handleChange(f.key, isChecked ? '0' : '1')}
              >
                <XStack space="$2" style={{ alignItems: 'center' }}>
                  <Checkbox checked={isChecked} size="$3">
                    <Checkbox.Indicator>
                      <Check/>
                    </Checkbox.Indicator>
                  </Checkbox>
                  <Label>{f.label}</Label>
                </XStack>
              </TouchableOpacity>
            )
          })}
        </YStack>
      </Card>

      {/* 가족력 */}
      <Card bordered padded elevate>
        <YStack space="$4">
          <Text fontSize="$5" fontWeight="700">
            가족력
          </Text>
          {familyHistoryFields.map((f) => {
            const isChecked = data[f.key] === '1'
            return (
              <TouchableOpacity
                key={f.key}
                onPress={() => handleChange(f.key, isChecked ? '0' : '1')}
              >
                <XStack space="$2" style={{ alignItems: 'center' }}>
                  <Checkbox checked={isChecked} size="$3">
                    <Checkbox.Indicator>
                      <Check/>
                    </Checkbox.Indicator>
                  </Checkbox>
                  <Label>{f.label}</Label>
                </XStack>
              </TouchableOpacity>
            )
          })}
        </YStack>
      </Card>

      {/* 라디오 그룹 */}
      {radioGroups.map((group) => (
        <Card key={group.key} bordered padded elevate>
          <YStack space="$3">
            <Text fontSize="$5" fontWeight="700">
              {group.label}
            </Text>
            <RadioGroup
              value={data[group.key] || ''}
              onValueChange={(val) => handleChange(group.key, val)}
            >
              <YStack space="$0.5">
                {group.options.map((opt) => (
                  <TouchableOpacity
                    key={opt.value}
                    onPress={() => handleChange(group.key, opt.value)}
                  >
                    <XStack space="$2" style={{ alignItems: 'center' }}>
                      <RadioGroup.Item value={opt.value} size="$3">
                        <RadioGroup.Indicator />
                      </RadioGroup.Item>
                      <Label>{opt.label}</Label>
                    </XStack>
                  </TouchableOpacity>
                ))}
              </YStack>
            </RadioGroup>
          </YStack>
        </Card>
      ))}
    </YStack>
  )
}
