INSTALL_HOME = $(shell pwd)

package :
	pip download -r $(INSTALL_HOME)/requirements.txt -d $(INSTALL_HOME)/requirements
	zip -r $(INSTALL_HOME)/hongsi_deploy.zip ./*

install : $(INSTALL_HOME)/requirements.txt $(INSTALL_HOME)/requirements $(INSTALL_HOME)/src/config/config.py
	pip install --ignore-installed --no-index --find-links=$(INSTALL_HOME)/requirements -r $(INSTALL_HOME)/requirements.txt
	sed -i "s?^ROOT_PATH.*?ROOT_PATH = '$(INSTALL_HOME)'?g" $(INSTALL_HOME)/src/config/config.py
run :
	export PYTHONPATH=$(PYTHONPATH):$(INSTALL_HOME) && \
	python3 $(INSTALL_HOME)/src/model_controller.py >> model_service.log 2>&1 & echo $$! > pid.txt

stat :
	ps aux | grep `cat pid.txt`

stop :
	kill -9 `cat pid.txt`
	rm pid.txt

clean :
	pip uninstall -yr $(INSTALL_HOME)/requirements.txt
