import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { HomeScreen } from '../screens/HomeScreen';
import { ResultsScreen } from '../screens/ResultsScreen';
import type { IdentifyResponse, SearchResponse } from '../services/api';

export type RootStackParamList = {
  Home: undefined;
  Results: {
    data: (IdentifyResponse | SearchResponse) & { recommendations: IdentifyResponse['recommendations'] };
    mode: 'identify' | 'search';
    query?: string;
  };
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export const AppNavigator: React.FC = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: '#080808' },
          animation: 'slide_from_bottom',
          animationDuration: 350,
        }}
      >
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="Results" component={ResultsScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};
