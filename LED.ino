#include <FastLED.h>

FASTLED_USING_NAMESPACE


#define DATA_PIN    7
//#define CLK_PIN   4
#define LED_TYPE    WS2811
#define COLOR_ORDER GRB
#define NUM_LEDS    48
CRGB leds[NUM_LEDS];

#define BRIGHTNESS          255
#define FRAMES_PER_SECOND  120

const int ledPin = 2;
char input;


bool On;
bool task;
int firstDebounce = 0;
int maintaining = 0;
int down = 0;
int downUnderThresh = 0;
int redLevel = 0;
int blueLevel = 250;
int brightness = 255;
int tmpRed = 0;
int tmpBlue = 0;
int taskTime = 0;


void setup() {
  delay(3000);
  FastLED.addLeds<LED_TYPE,DATA_PIN,COLOR_ORDER>(leds, NUM_LEDS).setCorrection(TypicalLEDStrip);
  FastLED.setBrightness(BRIGHTNESS);
  fill_solid(leds, NUM_LEDS, CRGB::Black);
  FastLED.show(); 
  
  




  
  // put your setup code here, to run once:

Serial.begin(9600);

}

void loop() {


  if(Serial.available() > 0){

//    fadeOut();
    
    input = Serial.read();

    char in = (char)input;
    Serial.flush();
    
  
    if(in == 'h'){
      if (millis() - firstDebounce>300){
        
 
//          On = true;
          tmpRed = redLevel;
          tmpBlue = blueLevel;
          if(tmpBlue != 0){          
          fill_solid(leds, NUM_LEDS, CRGB(tmpRed,0,tmpBlue));
          FastLED.show();
          }else{
          fill_solid(leds, NUM_LEDS, CRGB(tmpRed, 0, 255));
          FastLED.show();
            }
          brightness = 255;
          task = true;
          firstDebounce = millis();
        }

      
    }else if(in == 't'){
//      Serial.flush();
      if(On){On = false;}
      if(millis() -  taskTime > 300){
           
        if(blueLevel >= 10){
          blueLevel -= 10;
          redLevel += 10;
          tmpRed = redLevel;
          tmpBlue = blueLevel;
          fill_solid(leds, NUM_LEDS, CRGB(redLevel, 0, blueLevel));
          FastLED.show(); 
//          DmxSimple.write(3,blueLevel);
//          DmxSimple.write(1, redLevel);
          }else{
            fill_solid(leds, NUM_LEDS, CRGB(255, 0, 0));
            FastLED.show(); 
//            DmxSimple.write(1,255);
            tmpRed = 255;
          }

        task = true;
        taskTime = millis();
      }
      
      }else if(in == 'm'){
//        task = false;
        
        if(tmpBlue == 0){
            tmpBlue = 250;
         }
         
//        Serial.flush();
//        if(millis() - maintaining > 300){
//          task = false;
          
          fill_solid(leds, NUM_LEDS, CRGB(redLevel, 0, blueLevel));
          FastLED.show(); 
//          DmxSimple.write(1, redLevel);
//          DmxSimple.write(3,blueLevel);
          task = true;
          maintaining = millis();
//        }

    
        
      }else if(in == 'p'){
//        Serial.flush();
        if(millis() - down > 300){
          if(tmpBlue == 0){
            tmpBlue = 250;
            }
          task = true;
          
          down = millis();
          }

      }
      
  }else{
    
  
    if(On){
       
       brightness = brightness - 1;
       fill_solid(leds, NUM_LEDS, CRGB(0, 0, brightness));
       FastLED.show(); 
//       DmxSimple.write(3, brightness);
       delay(10);
       if(brightness == 0){
       On = false;
       }    
    }
    else if(task){
      if(blueLevel > redLevel){
        tmpBlue--;
        tmpRed = map(tmpBlue,0,blueLevel, 0,redLevel);
      }else{
        tmpRed--;
        tmpBlue = map(tmpRed,0,redLevel, 0,blueLevel);
        }
            fill_solid(leds, NUM_LEDS, CRGB(tmpRed, 0, tmpBlue));
            
//            DmxSimple.write(3,tmpBlue);
//            DmxSimple.write(1,tmpRed);
            delay(10);
            FastLED.show(); 

//        Serial.println(tmpBlue);
        if(tmpBlue == 0 && tmpRed == 0){
          task = false;

          }

      
    }
  }
}
