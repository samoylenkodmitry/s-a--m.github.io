---
layout: leetcode-entry
title: "1603. Design Parking System"
permalink: "/leetcode/problem/2023-05-29-1603-design-parking-system/"
leetcode_ui: true
entry_slug: "2023-05-29-1603-design-parking-system"
---

[1603. Design Parking System](https://leetcode.com/problems/design-parking-system/description/) easy
[blog post](https://leetcode.com/problems/design-parking-system/solutions/3573683/kotlin/)
[substack](https://dmitriisamoilenko.substack.com/p/27052023-1603-design-parking-system?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/227
#### Problem TLDR
Return if car of type `1, 2 or 3` can be added given sizes `big, medium and small`.
#### Intuition
Just write the code.

#### Approach
Let's use an array to minimize the number of lines.
#### Complexity
- Time complexity:
$$O(1)$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

class ParkingSystem(big: Int, medium: Int, small: Int) {
    val types = arrayOf(big, medium, small)

    fun addCar(carType: Int): Boolean = types[carType - 1]-- > 0
}

```

