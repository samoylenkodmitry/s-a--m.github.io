---
layout: leetcode-entry
title: "2353. Design a Food Rating System"
permalink: "/leetcode/problem/2023-12-17-2353-design-a-food-rating-system/"
leetcode_ui: true
entry_slug: "2023-12-17-2353-design-a-food-rating-system"
---

[2353. Design a Food Rating System](https://leetcode.com/problems/design-a-food-rating-system/description/) medium
[blog post](https://leetcode.com/problems/design-a-food-rating-system/solutions/4415744/kotlin-treeset/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17122023-2353-design-a-food-rating?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
![image.png](/assets/leetcode_daily_images/b5d8c549.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/442

#### Problem TLDR

Given foods, cuisines and ratings implement efficient methods changeRating(food, newRating) and highestRated(cuisine)

#### Intuition

Given that we must maintain sorted order by `rating` and be able to change the rating, the `TreeSet` may help, as it provides O(logN) amortized time for `remove(obj)`.

#### Approach

Start with inefficient implementation, like do the linear search in both methods. Then decide what data structures can help to quickly find an item.

* keep in mind, that `constructor` should also be efficient

#### Complexity

- Time complexity:
$$O(log(n))$$ for either method

- Space complexity:
$$O(n)$$

#### Code

```kotlin
class FoodRatings(val foods: Array<String>, val cuisines: Array<String>, val ratings: IntArray) {
  val foodToInd = foods.indices.groupBy { foods[it] }
  val cuisineToInds: MutableMap<String, TreeSet<Int>> = mutableMapOf()
  init {
    for (ind in cuisines.indices)
      cuisineToInds.getOrPut(cuisines[ind]) {
        TreeSet(compareBy({ -ratings[it] }, { foods[it] }))
      } += ind
  }

  fun changeRating(food: String, newRating: Int) {
    val ind = foodToInd[food]!![0]
    if (ratings[ind] != newRating) {
      val sortedInds = cuisineToInds[cuisines[ind]]!!
      sortedInds.remove(ind)
      ratings[ind] = newRating
      sortedInds.add(ind)
    }
  }

  fun highestRated(cuisine: String): String = foods[cuisineToInds[cuisine]!!.first()]
}

```

