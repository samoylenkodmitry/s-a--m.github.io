---
layout: leetcode-entry
title: "706. Design HashMap"
permalink: "/leetcode/problem/2023-10-04-706-design-hashmap/"
leetcode_ui: true
entry_slug: "2023-10-04-706-design-hashmap"
---

[706. Design HashMap](https://leetcode.com/problems/design-hashmap/description/) easy
[blog post](https://leetcode.com/problems/design-hashmap/solutions/4127340/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/4102023-706-design-hashmap?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/f2c844a4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/359

#### Problem TLDR

Design a HashMap

#### Intuition

The simple implementation consists of a growing array of buckets, where each bucket is a list of key-value pairs.

#### Approach

For better performance:
* use `LinkedList`
* start with smaller buckets size

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$, for all operations

#### Code

```kotlin

class MyHashMap() {
    var table = Array<MutableList<Pair<Int, Int>>>(16) { mutableListOf() }
    var count = 0

    fun bucket(key: Int) = table[key % table.size]

    fun rehash() = with(table.flatMap { it }) {
      table = Array(table.size * 2) { mutableListOf() }
      for ((key, value) in this) bucket(key) += key to value
    }

    fun put(key: Int, value: Int) = with(bucket(key)) {
      if (removeAll { it.first == key }) count++
      this += key to value
      if (count > table.size) rehash()
    }

    fun get(key: Int) = bucket(key)
      .firstOrNull { it.first == key }?.second ?: -1

    fun remove(key: Int) {
      if (bucket(key).removeAll { it.first == key }) count--
    }
}

```

