---
layout: leetcode-entry
title: "950. Reveal Cards In Increasing Order"
permalink: "/leetcode/problem/2024-04-10-950-reveal-cards-in-increasing-order/"
leetcode_ui: true
entry_slug: "2024-04-10-950-reveal-cards-in-increasing-order"
---

[950. Reveal Cards In Increasing Order](https://leetcode.com/problems/reveal-cards-in-increasing-order/description/) medium
[blog post](https://leetcode.com/problems/reveal-cards-in-increasing-order/solutions/5002042/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10042024-950-reveal-cards-in-increasing?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/g1AtbyjjmhU)
![2024-04-10_09-01.webp](/assets/leetcode_daily_images/dc55b3fa.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/567

#### Problem TLDR

Sort cards by rules: take top, next goes bottom #medium

#### Intuition

Let's reverse the problem: go from the last number, then prepend a value and rotate.

#### Approach

We can use `ArrayDeque` in Kotlin and just a `vec[]` in Rust (however `VecDeque` is also handy and make O(1) operation instead of O(n)).

#### Complexity

- Time complexity:
$$O(nlogn)$$, O(n^2) for vec[] solution, but the real time is still 0ms.

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun deckRevealedIncreasing(deck: IntArray) = with(ArrayDeque<Int>()) {
        deck.sortDescending()
        for (n in deck) {
            if (size > 0) addFirst(removeLast())
            addFirst(n)
        }
        toIntArray()
    }

```
```rust

    pub fn deck_revealed_increasing(mut deck: Vec<i32>) -> Vec<i32> {
        deck.sort_unstable_by_key(|n| -n);
        let mut queue = vec![];
        for n in deck {
            if queue.len() > 0 { queue.rotate_right(1) }
            queue.insert(0, n)
        }
        queue
    }

```

