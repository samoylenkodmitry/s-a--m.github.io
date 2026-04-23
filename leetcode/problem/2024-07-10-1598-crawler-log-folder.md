---
layout: leetcode-entry
title: "1598. Crawler Log Folder"
permalink: "/leetcode/problem/2024-07-10-1598-crawler-log-folder/"
leetcode_ui: true
entry_slug: "2024-07-10-1598-crawler-log-folder"
---

[1598. Crawler Log Folder](https://leetcode.com/problems/crawler-log-folder/description/) easy
[blog post](https://leetcode.com/problems/crawler-log-folder/solutions/5451533/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10072024-1598-crawler-log-folder?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/RUzb5ZbMcPw)
![2024-07-10_07-40_1.webp](/assets/leetcode_daily_images/c707f359.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/665

#### Problem TLDR

Filesystem path depth #easy

#### Intuition

Simulate the process and compute depth. Corner case: a path doesn't move up from the root.

#### Approach

Let's use `fold` iterator.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minOperations(logs: Array<String>) =
        logs.fold(0) { depth, op -> when (op) {
            "../" -> max(0, depth - 1)
            "./" -> depth else -> depth + 1 }}

```
```rust

    pub fn min_operations(logs: Vec<String>) -> i32 {
        logs.iter().fold(0, |depth, op| match op.as_str() {
            "../" => 0.max(depth - 1),
            "./" => depth,  _ =>  depth + 1 })
    }

```

