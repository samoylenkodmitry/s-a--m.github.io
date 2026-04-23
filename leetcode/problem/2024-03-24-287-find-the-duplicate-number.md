---
layout: leetcode-entry
title: "287. Find the Duplicate Number"
permalink: "/leetcode/problem/2024-03-24-287-find-the-duplicate-number/"
leetcode_ui: true
entry_slug: "2024-03-24-287-find-the-duplicate-number"
---

[287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/description/) medium
[blog post](https://leetcode.com/problems/find-the-duplicate-number/solutions/4918291/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24032024-287-find-the-duplicate-number?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/XFLC-YG7N14)
![2024-03-24_11-13_1.webp](/assets/leetcode_daily_images/dd2976c8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/548

#### Problem TLDR

Duplicate single number in `1..n` array, no extra memory #medium

# Intuition

The idea of existing cycle would come to mind after some hitting your head against the wall. The interesting fact is we must find the node that is not a port of the cycle: so the meeting point will be our answer:
![2024-03-24_10-35.jpg](/assets/leetcode_daily_images/55567c07.webp)
Now the clever trick is we can treat `node 0` as this external node:
![2024-03-24_10-55.jpg](/assets/leetcode_daily_images/4262a17f.webp)
This will coincidentally make our code much cleaner, I think this was the intention of the question authors.

#### Approach

Draw some circles and arrows, walk the algorithm with your hands.
To find the meeting point you must reset one pointer to the start.
* The Rust's `do-while-do` loop is perfectly legal https://programming-idioms.org/idiom/78/do-while-loop/795/rust

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun findDuplicate(nums: IntArray): Int {
    var fast = 0; var slow = 0
    do {
      fast = nums[nums[fast]]
      slow = nums[slow]
    } while (fast != slow)
    slow = 0
    while (fast != slow) {
      fast = nums[fast]
      slow = nums[slow]
    }
    return slow
  }

```
```rust

  pub fn find_duplicate(nums: Vec<i32>) -> i32 {
    let (mut tortoise, mut hare) = (0, 0);
    while {
      hare = nums[nums[hare as usize] as usize];
      tortoise = nums[tortoise as usize];
      hare != tortoise
    }{}
    hare = 0;
    while (hare != tortoise) {
      hare = nums[hare as usize];
      tortoise = nums[tortoise as usize]
    }
    tortoise
  }

```

