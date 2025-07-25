import React from 'react';
import { View, Button, Text } from 'react-native';
import axios from 'axios';

export default function SubmitButton({
  questionnaire,
  ecgImage,
  setMessage,
  setLoading,
  loading,
  message,
}: {
  questionnaire: Record<string, string>,
  ecgImage: any,
  setMessage: (msg: string) => void,
  setLoading: (b: boolean) => void,
  loading: boolean,
  message: string
}) {

  const handleSubmit = async () => {
    if (!ecgImage) return setMessage('❗ ECG 이미지를 업로드해주세요.');
    if (!questionnaire.age || !questionnaire.weight || !questionnaire.height) {
      return setMessage('❗ 모든 항목을 입력해주세요.');
    }

    setLoading(true);
    setMessage('');

    const formData = new FormData();
    formData.append('file', {
      uri: ecgImage.uri,
      name: 'ecg.jpg',
      type: 'image/jpeg',
    } as any);

    Object.entries(questionnaire).forEach(([k, v]) => {
      formData.append(k, v);
    });

    try {
      const res = await axios.post('http://<서버-IP>:8080/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setMessage('✅ 업로드 성공');
    } catch (err) {
      console.error(err);
      setMessage('❌ 서버 오류 발생');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View>
      <Button title={loading ? '업로드 중...' : '제출'} onPress={handleSubmit} disabled={loading} />
      {message && (
        <Text className='mt-2 text-center text-red-500 font-medium'>{message}</Text>
      )}
    </View>
  );
}
