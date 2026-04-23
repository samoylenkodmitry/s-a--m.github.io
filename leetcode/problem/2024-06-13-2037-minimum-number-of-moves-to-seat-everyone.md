---
layout: leetcode-entry
title: "2037. Minimum Number of Moves to Seat Everyone"
permalink: "/leetcode/problem/2024-06-13-2037-minimum-number-of-moves-to-seat-everyone/"
leetcode_ui: true
entry_slug: "2024-06-13-2037-minimum-number-of-moves-to-seat-everyone"
---

[2037. Minimum Number of Moves to Seat Everyone](https://leetcode.com/problems/minimum-number-of-moves-to-seat-everyone/description/) easy
[blog post](https://leetcode.com/problems/minimum-number-of-moves-to-seat-everyone/solutions/5304834/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13062024-2037-minimum-number-of-moves?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Fo-myk0njiU)
![2024-06-13_06-42_1.webp](/assets/leetcode_daily_images/3295fa23.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/638

#### Problem TLDR

Sum of diffs of sorted students and seats #easy

#### Intuition

Deduce the intuition from the problem examples: the optimal solution is to take difference between sorted seats and students greedily.

#### Approach

Let's use some languages iterators:
* Kotlin: `sorted`, `zip`, `sumOf`
* Rust: `iter`, `zip`, `sum`

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$ for Kotlin, O(1) for Rust solution

#### Code

```kotlin

    fun minMovesToSeat(seats: IntArray, students: IntArray) =
        seats.sorted().zip(students.sorted()).sumOf { (a, b) -> abs(a - b) }

```
```rust

    pub fn min_moves_to_seat(mut seats: Vec<i32>, mut students: Vec<i32>) -> i32 {
        seats.sort_unstable(); students.sort_unstable();
        seats.iter().zip(students).map(|(a, b)| (a - b).abs()).sum()
    }

```

