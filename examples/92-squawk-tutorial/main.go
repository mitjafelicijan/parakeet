package main

import (
	"fmt"

	"github.com/mitjafelicijan/parakeet/enums/option"
	"github.com/mitjafelicijan/parakeet/enums/provider"
	"github.com/mitjafelicijan/parakeet/llm"
	"github.com/mitjafelicijan/parakeet/squawk"
)

func main() {
	ollamaBaseUrl := "http://localhost:11434"
	model := "qwen2.5:1.5b"

	options := llm.SetOptions(map[string]interface{}{
		option.Temperature:   0.5,
		option.RepeatLastN:   2,
		option.RepeatPenalty: 2.2,
	})

	squawk.New().
		Model(model).
		BaseURL(ollamaBaseUrl).
		Provider(provider.Ollama).
		Options(options).
		System("You are a useful AI agent, you are a Star Trek expert.").
		User("Who is James T Kirk?").
		Chat(func(answer llm.Answer, self *squawk.Squawk, err error) {
			fmt.Println(answer.Message.Content)
		}).
		SaveAssistantAnswer().
		User("Who is his best friend?").
		Chat(func(answer llm.Answer, self *squawk.Squawk, err error) {
			fmt.Println(answer.Message.Content)
		})

}
