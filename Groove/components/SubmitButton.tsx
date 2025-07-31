import React from 'react'
import { Button, Spinner, Text, YStack, XStack } from 'tamagui'
import axios from 'axios'
import { useNavigation } from '@react-navigation/native'
import type { NativeStackNavigationProp } from '@react-navigation/native-stack'
import { useRouter } from 'expo-router'


export default function SubmitButton({
  questionnaire,
  ecgImage,
  setMessage,
  setLoading,
  loading,
  message,
}: {
  questionnaire: Record<string, string>
  ecgImage: any
  setMessage: (msg: string) => void
  setLoading: (b: boolean) => void
  loading: boolean
  message: string
}) {
  const router = useRouter()

  const handleSubmit = async () => {
    if (!ecgImage) return setMessage('❗ ECG 이미지를 업로드해주세요.')
    if (!questionnaire.age || !questionnaire.weight || !questionnaire.height) {
      return setMessage('❗ 모든 항목을 입력해주세요.')
    }

    setLoading(true)
    setMessage('')

    const formData = new FormData()
    formData.append('file', {
      uri: ecgImage.uri,
      name: ecgImage.fileName ?? 'photo.jpg',
      type: ecgImage.mimeType ?? 'image/jpeg',
    } as any)
    formData.append('questionnaire', JSON.stringify(questionnaire))

    try {
      const res = await axios.post(
        'http://172.20.10.7:8080/api/public/upload-ecgImage-lead2only',
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
      if (res.status === 200 && res.data) {
        router.push({
        pathname: '/ResultScreen',
        params: { result: JSON.stringify(res.data) },
      })
      } else {
        setMessage('❌ 결과를 불러올 수 없습니다.')
      }
    } catch (err) {
      console.error(err)
      setMessage('❌ 서버 오류 발생')
    } finally {
      setLoading(false)
    }
  }

  return (
    <YStack space="$2" style={{ alignItems: 'center' }}>
      <Button onPress={handleSubmit} disabled={loading}>
        {loading ? (
          <XStack space="$2" style={{ alignItems: 'center'}}>
            <Spinner size="small" />
            <Text>업로드 중...</Text>
          </XStack>
        ) : (
          <Text>제출</Text>
        )}
      </Button>
      {!!message && (
        <Text color="$red10" fontWeight="600" style={{ textAlign: 'center' }}>
          {message}
        </Text>
      )}
    </YStack>
  )
}
