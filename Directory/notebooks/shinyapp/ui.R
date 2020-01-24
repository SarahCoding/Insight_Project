library(shiny)

# Define UI for miles per gallon application
shinyUI(pageWithSidebar(
  
  # Application title
  headerPanel("Twitter Speech Reaction"),
  
  # Sidebar with controls to select the variable to plot against mpg
  # and to specify whether outliers should be included
  sidebarPanel(
    selectInput("variable", "Variable:",
                list("Bernie Sanders" = "BS")),
    
    checkboxInput("outliers", "Show outliers", FALSE)
  ),
  
  mainPanel()
))