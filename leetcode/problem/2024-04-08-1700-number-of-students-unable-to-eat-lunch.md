---
layout: leetcode-entry
title: "1700. Number of Students Unable to Eat Lunch"
permalink: "/leetcode/problem/2024-04-08-1700-number-of-students-unable-to-eat-lunch/"
leetcode_ui: true
entry_slug: "2024-04-08-1700-number-of-students-unable-to-eat-lunch"
---

[1700. Number of Students Unable to Eat Lunch](https://leetcode.com/problems/number-of-students-unable-to-eat-lunch/description/) easy
[blog post](https://leetcode.com/problems/number-of-students-unable-to-eat-lunch/solutions/4991239/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08042024-1700-number-of-students?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/WzOrYzC3UbM)
![2024-04-08_08-24.webp](/assets/leetcode_daily_images/c1fe52be.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/564

#### Problem TLDR

First sandwitch not eaten by any while popped from a queue  #easy

#### Intuition

First, understant the problem: we searching the first `sandwitch` which none of the students are able to eat.
The simulation code is straighforward and takes O(n^2) time which is accepted.
However, we can count how many students are `0`-eaters and how many `1`-eaters, then stop when none are able to eat current sandwitch.

#### Approach

We can use two counters or one array. How many lines of code can you save?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun countStudents(students: IntArray, sandwiches: IntArray): Int {
        val count = IntArray(2)
        for (s in students) count[s]++
        for ((i, s) in sandwiches.withIndex())
            if (--count[s] < 0) return students.size - i
        return 0
    }

```
```rust

    pub fn count_students(students: Vec<i32>, sandwiches: Vec<i32>) -> i32 {
        let (mut count, n) = (vec![0; 2], students.len());
        for s in students { count[s as usize] += 1 }
        for (i, &s) in sandwiches.iter().enumerate() {
            count[s as usize] -= 1;
            if count[s as usize] < 0 { return (n - i) as i32 }
        }; 0
    }

```

