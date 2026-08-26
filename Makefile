IMAGE := cs683-x86
PLATFORM := linux/amd64

.PHONY: docker-build docker-run docker-clean

docker-build:
	docker build --platform $(PLATFORM) -t $(IMAGE) .

docker-run:
	docker run --rm -it \
		--platform $(PLATFORM) \
		-v "$(PWD):/workspace" \
		-w /workspace \
		$(IMAGE)

docker-clean:
	docker rmi $(IMAGE)