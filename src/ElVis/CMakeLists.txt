SUBDIRS(
    Core
	# Extensions must come before ElVisCLI since that is where the tests are setup.
    Extensions
    ElVisCLI
    Gui
    ModelConversion
    QtPropertyBrowser
    #tinyxml
)

