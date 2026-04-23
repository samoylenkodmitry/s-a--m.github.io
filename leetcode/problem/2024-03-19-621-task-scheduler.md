---
layout: leetcode-entry
title: "621. Task Scheduler"
permalink: "/leetcode/problem/2024-03-19-621-task-scheduler/"
leetcode_ui: true
entry_slug: "2024-03-19-621-task-scheduler"
---

[621. Task Scheduler](https://leetcode.com/problems/task-scheduler/description/) medium
[blog post](https://leetcode.com/problems/task-scheduler/solutions/4895943/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19032024-621-task-scheduler?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/8t1KNa9iZjA)
![2024-03-19_10-10.jpg](/assets/leetcode_daily_images/115cd81a.webp)
https://youtu.be/8t1KNa9iZjA
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/543

#### Problem TLDR

Count CPU cycles if task can't run twice in `n` cycles #medium

#### Intuition
Let's try to understand the problem first, by observing the example:
```j
    // 0 1 2 3 4 5 6 7
    // a a a b b b c d n = 3
    // a . . . a . . . a
    //   b . . . b . . . b
    //     c d     i i
```
One inefficient way is to take tasks by thier frequency, store availability and adjust cycle forward if no task available. This solution will take O(n) time but with big constant of iterating and sorting the frequencies `[26]` array.

The clever way is to notice the pattern of how tasks are: there are empty slots between the most frequent task(s).

#### Approach

In the interview I would choose the first way.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun leastInterval(tasks: CharArray, n: Int): Int {
    val f = IntArray(128); for (c in tasks) f[c.code]++
    val maxFreq = f.max()
    val countOfMaxFreq = f.count { it == maxFreq }
    val slotSize = n - (countOfMaxFreq - 1)
    val slotsCount = (maxFreq - 1) * slotSize
    val otherTasks = tasks.size - maxFreq * countOfMaxFreq
    val idles = max(0, slotsCount - otherTasks)
    return tasks.size + idles
  }

```
```rust

    pub fn least_interval(tasks: Vec<char>, n: i32) -> i32 {
      let mut f = vec![0; 128]; for &c in &tasks { f[c as usize] += 1 }
      let maxFreq = f.iter().max().unwrap();
      let countOfMaxFreq = f.iter().filter(|&x| x == maxFreq).count() as i32;
      let slotsCount = (maxFreq - 1) * (n - countOfMaxFreq + 1);
      let otherTasks = tasks.len() as i32 - maxFreq * countOfMaxFreq;
      tasks.len() as i32 + (slotsCount - otherTasks).max(0)
    }

```

