.PHONY: test
test:
	cargo test

.PHONY: check
check:
	cargo check

.PHONY: lint
lint:
	cargo clippy -- -D warnings

.PHONY: doc
doc:
	cargo doc --open

.PHONY: cat
cat:
	cargo run --example cat

.PHONY: checkerboard
checkerboard:
	cargo run --example checkerboard

.PHONY: build-shaders
build-shaders:
	./src/backends/shaders/compile.sh
