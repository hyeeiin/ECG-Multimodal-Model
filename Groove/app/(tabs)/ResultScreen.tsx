import React, { useState, useLayoutEffect, useRef } from 'react'
import {
  Dimensions, ScrollView, View
} from 'react-native'
import { useLocalSearchParams } from 'expo-router'
import {
  YStack,
  Text,
  Image,
  Button,
  Card,
} from 'tamagui'
import { LineChart, PieChart } from 'react-native-chart-kit'
import { useNavigation } from "@react-navigation/native"
import { captureRef } from 'react-native-view-shot'
import * as MediaLibrary from 'expo-media-library'
import * as FileSystem from 'expo-file-system'
import * as Sharing from 'expo-sharing'

export default function ResultScreen() {
  const { result } = useLocalSearchParams()
  const parsedResult = JSON.parse(result as string)

  const [showHeatmap, setShowHeatmap] = useState(false)

  const {
    label,
    probability,
    ecg_signal,
    heatmap,
    feature_importance,
    gpt_result,
    pwv_shap_report,
    pwv_shap_img_base64,
  } = parsedResult

  const screenWidth = Dimensions.get('window').width
  const chartHeight = 180

  const chartData = {
    labels: [],
    datasets: [
      {
        data: ecg_signal.map((d: any) => d['Voltage (mV)']),
        strokeWidth: 2,
        color: () => '#000',
      },
    ],
  }

  const pieData = [
    { name: 'Image', value: Math.round(feature_importance.image), color: '#facc15', legendFontColor: '#000', legendFontSize: 12 },
    { name: 'Signal', value: Math.round(feature_importance.signal), color: '#4ade80', legendFontColor: '#000', legendFontSize: 12 },
    { name: 'Age', value: Math.round(feature_importance.age), color: '#60a5fa', legendFontColor: '#000', legendFontSize: 12 },
    { name: 'Weight', value: Math.round(feature_importance.wt), color: '#c084fc', legendFontColor: '#000', legendFontSize: 12 },
  ]

  const navigation = useNavigation()
  useLayoutEffect(() => {
    navigation.setOptions({
      title: "분석 결과", // ← 원하는 텍스트로 설정
    })
  }, [navigation])

   const viewRef = useRef(null)

  const handleCapture = async () => {
    try {
      const uri = await captureRef(viewRef, {
        format: 'png',
        quality: 1,
      })

      const permission = await MediaLibrary.requestPermissionsAsync()
      if (permission.granted) {
        const asset = await MediaLibrary.createAssetAsync(uri)
        await MediaLibrary.createAlbumAsync('ECG_Results', asset, false)
        alert('✅ 이미지가 저장되었습니다!')
      } else {
        alert('❌ 저장 권한이 필요합니다.')
      }

      // Optional: 공유 기능
      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(uri)
      }

    } catch (err) {
      console.error('이미지 저장 실패:', err)
    }
  }

  return (
    <ScrollView ref={viewRef} style={{padding: 16, backgroundColor: 'white'}} contentContainerStyle={{ flexGrow: 1 }}>
        {/* <Card padded elevation="$2" mb="$4">
          <YStack space>
            <Text fontSize="$6" fontWeight="600">📊 ECG 파형</Text>

            {(
              <YStack width="100%" height={chartHeight} z={3}>
                <Image
                  source={{ uri: `data:image/png;base64,${heatmap}` }}
                  width="100%"
                  height={chartHeight}
                  resizeMode="cover"
                  opacity={0.3}
                />
              </YStack>
            )}
          </YStack>
        </Card> */}

        <Card padded elevation="$2" mb="$4">
          <YStack space>
            <Text fontSize="$6" fontWeight="600">📊 ECG 파형</Text>

            <ScrollView horizontal showsHorizontalScrollIndicator>
              <YStack width={screenWidth * 2} height={chartHeight}>
                {/* heatmap을 배경으로 */}
                {showHeatmap && (
                <Image
                  source={{ uri: `data:image/png;base64,${heatmap}` }}
                  style={{
                    position: 'absolute',
                    width: screenWidth * 2 - 64,
                    height: chartHeight,
                    opacity: 0.3,
                    zIndex: 3,
                    top: 0,
                    left: 64
                  }}
                  resizeMode="stretch" // signal과 정렬되도록 stretch
                />
                )}

                {/* signal 위에 그리기 */}
                <LineChart
                  data={chartData}
                  width={screenWidth * 2}
                  height={chartHeight}
                  withDots={false}
                  withShadow={false}
                  chartConfig={{
                    backgroundColor: '#fff',
                    backgroundGradientFrom: '#fff',
                    backgroundGradientTo: '#fff',
                    decimalPlaces: 3,
                    color: () => '#fff',
                    labelColor: () => '#000',
                    propsForBackgroundLines: { strokeDasharray: '' },
                  }}
                  style={{
                    borderRadius: 12,
                    zIndex: 2,
                  }}
                />
              </YStack>
            </ScrollView>

            <Button
              onPress={() => setShowHeatmap((prev) => !prev)}
              themeInverse
              size="$3"
              mt="$2"
            >
              {showHeatmap ? '🔻 주요 영역 끄기' : '🔺 주요 영역 보기'}
            </Button>
          </YStack>
        </Card>

        {/* 📌 진단 요약 + 중요도 Pie */}
        <Card padded elevation="$2" mb="$4">
          <YStack space="$3">
            <Text fontSize="$6" fontWeight="600">📌 진단 요약</Text>
            <Text>결과: {label === 1 ? '이상 (Abnormal)' : '정상 (Normal)'}</Text>
            <Text>모델 예측 확률: {(probability * 100).toFixed(1)}%</Text>
            <Text>해당 결과는 다음 항목들의 중요도를 기반으로 판단되었습니다.</Text>

            <PieChart
              data={pieData}
              width={screenWidth - 64}
              height={180}
              accessor="value"
              chartConfig={{
                backgroundColor: '#fff',
                backgroundGradientFrom: '#fff',
                backgroundGradientTo: '#fff',
                color: () => '#000',
                labelColor: () => '#000',
              }}
              backgroundColor="transparent"
              paddingLeft="20"
              absolute
            />

            <Text color="$red10" mt="$2">
              ※ 중요도는 이미지, 신호, 나이, 체중에 각각 할당된 영향력을 기반으로 계산된 결과입니다.
            </Text>
          </YStack>
        </Card>

        {/* 🤖 GPT 해석 */}
        <Card padded elevation="$2" mb="$4">
          <YStack space>
            <Text fontSize="$6" fontWeight="600">🤖 GPT 기반 해석</Text>
            {gpt_result ? (
              Object.entries(gpt_result).map(([key, val]) => (
                <Card
                  key={key}
                  padded
                  bordered
                  elevate
                  mb="$2"
                  backgroundColor="$background"
                >
                  <Text fontWeight="700" mb="$1">🔹 {key}</Text>
                  <Text>{String(val)}</Text>
                </Card>
              ))
            ) : (
              <Text>해석 없음</Text>
            )}
          </YStack>
        </Card>
      <YStack space mb={25}>
        <Button onPress={handleCapture}>📸 결과 이미지 저장</Button>
      </YStack>
    </ScrollView>
  )
}
