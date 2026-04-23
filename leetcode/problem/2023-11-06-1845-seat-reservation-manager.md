---
layout: leetcode-entry
title: "1845. Seat Reservation Manager"
permalink: "/leetcode/problem/2023-11-06-1845-seat-reservation-manager/"
leetcode_ui: true
entry_slug: "2023-11-06-1845-seat-reservation-manager"
---

[1845. Seat Reservation Manager](https://leetcode.com/problems/seat-reservation-manager/description/) medium
[blog post](https://leetcode.com/problems/seat-reservation-manager/solutions/4255246/kotlin-priorityqueue/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06112023-1845-seat-reservation-manager?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/f1e44ac2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/395

#### Problem TLDR

Design reservation number system

#### Intuition

The naive approach is to just use PriorityQueue as is:

```kotlin
class SeatManager(n: Int): PriorityQueue<Int>() {
  init { for (x in 1..n) add(x) }
  fun reserve() = poll()
  fun unreserve(seatNumber: Int) = add(seatNumber)
}
```

However, we can improve the memory cost by noticing, that we can shrink the heap when `max` is returned.

#### Approach

* we can save some lines of code by using extending the class (prefer a field instead in a production code to not exprose the heap directly)

#### Complexity

- Time complexity:
$$O(log(n))$$ for operations

- Space complexity:
$$O(n)$$

#### Code

```kotlin

class SeatManager(n: Int): PriorityQueue<Int>() {
  var max = 0
  fun reserve() = if (isEmpty()) ++max else poll()
  fun unreserve(seatNumber: Int) {
    if (seatNumber == max) max--
    else add(seatNumber)
  }
}

```

