## Only run examples in interactive R sessions
if (interactive()) {
  
  # basic example
  shinyApp(
    ui = fluidPage(
      selectInput("variable", "Variable:",
                  c("Bernie Sanders" = "cyl",
                    "Transmission" = "am",
                    "Gears" = "gear")),
      tableOutput("data")
    ),
    server = function(input, output) {
      output$data <- renderTable({
        mtcars[, c("mpg", input$variable), drop = FALSE]
      }, rownames = TRUE)
    }
  )
  
  # demoing group support in the `choices` arg
  shinyApp(
    ui = fluidPage(
      selectInput("state", "Choose a state:",
                  list(`Candidate` = list("Hillary Clinton", "Bernie Sanders"))
      ),
      textOutput("result")
    ),
    server = function(input, output) {
      output$result <- renderText({
        paste("You chose", input$state)
      })
    }
  )
}
